
import json
import os
import shutil
import warnings
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any
from uuid import uuid4

import gradio as gr
import numpy as np
import requests
from filelock import FileLock
from PIL import Image, PngImagePlugin
import filetype
import wave
from mutagen.mp3 import MP3, EasyMP3
import torchaudio
import subprocess
from tqdm import tqdm
from io import BytesIO
from urllib.parse import urlparse
import requests

user_profile = gr.State(None)

def get_profile() -> gr.OAuthProfile | None:
    """
    Retrieve the currently logged-in user's profile.

    This function returns the user profile stored in the global Gradio state.
    If no user is logged in, it returns None.

    Returns:
        gr.OAuthProfile | None: The currently logged-in user's profile, or None if no user is logged in.
    """
    global user_profile
    """Return the user profile if logged in, None otherwise."""

    return user_profile

def setup(folder_path: str | Path | None = None, display_type: str = "image_path") -> None:
    user_history = _UserHistory()
    user_history.folder_path = _resolve_folder_path(folder_path)
    user_history.display_type = display_type
    user_history.initialized = True


def render() -> None:
    user_history = _UserHistory()

    # initialize with default config
    if not user_history.initialized:
        print("Initializing user history with default config. Use `user_history.setup(...)` to customize folder_path.")
        setup()

    # Render user history tab
    gr.Markdown(
        "## Your past generations\n\nLog in to keep a gallery of your previous generations. Your history will be saved"
        " and available on your next visit. Make sure to export your images from time to time as this gallery may be"
        " deleted in the future."
    )

    if os.getenv("SYSTEM") == "spaces" and not os.path.exists("/data"):
        gr.Markdown(
            "**⚠️ Persistent storage is disabled, meaning your history will be lost if the Space gets restarted."
            " Only the Space owner can setup a Persistent Storage. If you are not the Space owner, consider"
            " duplicating this Space to set your own storage.⚠️**"
        )

    with gr.Row():
        gr.LoginButton(min_width=250)
        #gr.LogoutButton(min_width=250)
        refresh_button = gr.Button(
            "Refresh",
            icon="./assets/icon_refresh.png",
        )
        export_button = gr.Button(
            "Export",
            icon="./assets/icon_download.png",
        )
        delete_button = gr.Button(
            "Delete history",
            icon="./assets/icon_delete.png",
        )

    # "Export zip" row (hidden by default)
    with gr.Row():
        export_file = gr.File(file_count="single", file_types=[".zip"], label="Exported history", visible=False)

    # "Config deletion" row (hidden by default)
    with gr.Row():
        confirm_button = gr.Button("Confirm delete all history", variant="stop", visible=False)
        cancel_button = gr.Button("Cancel", visible=False)

    # Gallery
    gallery = gr.Gallery(
        label="Past images",
        show_label=True,
        elem_id="gradio_user_history_gallery",
        object_fit="cover",
        columns=5,
        height=600,
        preview=False,
        show_share_button=True,
        show_download_button=True,
    )
    gr.Markdown(
        "User history is powered by"
        " [Wauplin/gradio-user-history](https://huggingface.co/spaces/Wauplin/gradio-user-history). Integrate it to"
        " your own Space in just a few lines of code!"
    )
    gallery.attach_load_event(_fetch_user_history, every=None)

    # Interactions
    refresh_button.click(fn=_fetch_user_history, inputs=[], outputs=[gallery], queue=False)
    export_button.click(fn=_export_user_history, inputs=[], outputs=[export_file], queue=False)

    # Taken from https://github.com/gradio-app/gradio/issues/3324#issuecomment-1446382045
    delete_button.click(
        lambda: [gr.update(visible=True), gr.update(visible=True)],
        outputs=[confirm_button, cancel_button],
        queue=False,
    )
    cancel_button.click(
        lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[confirm_button, cancel_button],
        queue=False,
    )
    confirm_button.click(_delete_user_history).then(
        lambda: [gr.update(visible=False), gr.update(visible=False)],
        outputs=[confirm_button, cancel_button],
        queue=False,
    )

    # Admin section (only shown locally or when logged in as Space owner)
    _admin_section()


def save_image(
    profile: gr.OAuthProfile | None,
    image: Image.Image | np.ndarray | str | Path,
    label: str | None = None,
    metadata: Dict | None = None,
):
    # Ignore images from logged out users
    if profile is None:
        return
    username = profile if isinstance(profile, str) else profile["preferred_username"]

    # Ignore images if user history not used
    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn(
            "User history is not set in Gradio demo. Saving image is ignored. You must use `user_history.render(...)`"
            " first."
        )
        return

    # Copy image to storage
    image_path = _copy_image(image, dst_folder=user_history._user_images_path(username))

    # Save new image + metadata
    if metadata is None:
        metadata = {}
    if "datetime" not in metadata:
        metadata["datetime"] = str(datetime.now())
    data = {"image_path": str(image_path), "label": label, "metadata": metadata}
    with user_history._user_lock(username):
        with user_history._user_jsonl_path(username).open("a") as f:
            f.write(json.dumps(data) + "\n")
            
def save_file(
    profile: gr.OAuthProfile | None,
    image: Image.Image | np.ndarray | str | Path | None = None,
    video: str | Path | None = None,
    audio: str | Path | None = None,
    document: str | Path | None = None,
    label: str | None = None,
    metadata: Dict | None = None,
    progress= gr.Progress(track_tqdm=True)
):
    # Ignore files from logged out users
    if profile is None:
        return
    username = profile if isinstance(profile, str) else profile["preferred_username"]

    # Ignore files if user history not used
    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn(
            "User history is not set in Gradio demo. Saving files is ignored. You must use `user_history.render(...)`"
            " first."
        )
        return

    uniqueId = uuid4().hex[:4]

    # Save new files + metadata
    if metadata is None:
        metadata = {}
    if "datetime" not in metadata:
        metadata["datetime"] = str(datetime.now())

    # Count operations to later update progress
    operations = []
    if audio is not None:
        operations.append("audio")
    if video is not None:
        operations.append("video")
    if image is not None:
        operations.append("image")
    if document is not None:
        operations.append("document")
    operations.append("jsonl")
    operations.append("cleanup")

    # Create a progress bar
    with tqdm(total=len(operations), desc="Saving files to history..") as pb:
        audio_path = None
        # Copy audio to storage
        if audio is not None:
            audio_path1 = _copy_file(audio, dst_folder=user_history._user_file_path(username, "audios"), uniqueId=uniqueId)
            audio_path = _add_metadata(audio_path1, metadata)
            pb.update(1)

        video_path = None
        # Copy video to storage - need audio_path if available
        if video is not None:
            video_path1 = _copy_file(video, dst_folder=user_history._user_file_path(username, "videos"), uniqueId=uniqueId)
            video_path = _add_metadata(video_path1, metadata, str(audio_path))
            pb.update(1)

        image_path = None
        # Copy image to storage - need video_path if available
        if image is not None:
            image_path1 = _copy_image(image, dst_folder=user_history._user_images_path(username), uniqueId=uniqueId)
            image_path = _add_metadata(image_path1, metadata)
            pb.update(1)

        document_path = None
        # Copy document to storage
        if document is not None:
            document_path1 = _copy_file(document, dst_folder=user_history._user_file_path(username, "documents"), uniqueId=uniqueId)
            document_path = _add_metadata(document_path1, metadata)
            pb.update(1)

        # Save Json file with combined data
        data = {
            "image_path": str(image_path),
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "document_path": str(document_path),
            "label": _UserHistory._sanitize_for_json(label),
            "metadata": _UserHistory._sanitize_for_json(metadata)
        }
        with user_history._user_lock(username):
            with user_history._user_jsonl_path(username).open("a") as f:
                f.write(json.dumps(data) + "\n")
        pb.update(1)

        # Cleanup
        if "audio" in operations and audio_path1 and audio_path1.exists():
            try:
                audio_path1.unlink()
            except Exception as e:
                print(f"An error occurred while deleting the audio history file: {e}")
        if "video" in operations and video_path1 and video_path1.exists():
            try:
                video_path1.unlink()
            except Exception as e:
                print(f"An error occurred while deleting the video history file: {e}")
        if "image" in operations and image_path1 and image_path1.exists():
            try:
                image_path1.unlink()
            except Exception as e:
                print(f"An error occurred while deleting the image history file: {e}")
        if "document" in operations and document_path1 and document_path1.exists():
            try:
                document_path1.unlink()
            except Exception as e:
                print(f"An error occurred while deleting the document history file: {e}")
        pb.update(1)

    # If no files were saved, nothing to do
    if image_path is None and video_path is None and audio_path is None and document_path is None:
        return
    # else:
    #     # Return the paths of the saved files
    #     return {
    #         "image_path": image_path,
    #         "video_path": video_path,
    #         "audio_path": audio_path,
    #         "document_path": document_path,
    #         "label": label,
    #         "metadata": metadata
    #     }, json.dumps(data)
    
def get_filepath():
    """Return the path to the user history folder."""
    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return None
    return user_history.folder_path

    

#############
# Internals #
#############


class _UserHistory(object):
    _instance = None
    initialized: bool = False
    folder_path: Path
    display_type: str = "video_path"

    def __new__(cls):
        # Using singleton pattern => we don't want to expose an object (more complex to use) but still want to keep
        # state between `render` and `save_image` calls.
        if cls._instance is None:
            cls._instance = super(_UserHistory, cls).__new__(cls)
        return cls._instance

    def _user_path(self, username: str) -> Path:
        path = self.folder_path / username
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _user_lock(self, username: str) -> FileLock:
        """Ensure history is not corrupted if concurrent calls."""
        return FileLock(self.folder_path / f"{username}.lock")  # lock outside of folder => better when exporting ZIP

    def _user_jsonl_path(self, username: str) -> Path:
        return self._user_path(username) / "history.jsonl"

    def _user_images_path(self, username: str) -> Path:
        path = self._user_path(username) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _user_file_path(self, username: str, filetype: str = "images") -> Path:
        path = self._user_path(username) / filetype
        path.mkdir(parents=True, exist_ok=True)
        return path
   
    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """
        Recursively convert non-serializable objects into their string representation.
        """
        if isinstance(obj, dict):
            return {str(key): _UserHistory._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_UserHistory._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif hasattr(obj, "isoformat"):
            # For datetime objects and similar.
            return obj.isoformat()
        else:
            return str(obj)
    

def _fetch_user_history(profile: gr.OAuthProfile | None) -> List[Tuple[str, str]]:
    """
    Return saved history for the given user.

    Args:
        profile (gr.OAuthProfile | None): The user profile.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the path to an image and its label.
    """
    # Cannot load history for logged out users
    global user_profile
    if profile is None:
        user_profile = gr.State(None)
        return []
    username = profile if isinstance(profile, str) else str(profile["preferred_username"])
    
    user_profile = gr.State(profile)

    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return []

    with user_history._user_lock(username):
        # No file => no history saved yet
        jsonl_path = user_history._user_jsonl_path(username)
        if not jsonl_path.is_file():
            return []

        # Read history
        images = []
        for line in jsonl_path.read_text().splitlines():
            data = json.loads(line)
            images.append((data[user_history.display_type], data["label"] or ""))
        return list(reversed(images))


def _export_user_history(profile: gr.OAuthProfile | None) -> Dict | None:
    """
    Zip all history for the given user, if it exists, and return it as a downloadable file.

    Args:
        profile (gr.OAuthProfile | None): The user profile.

    Returns:
        Dict | None: A Gradio update dictionary with the path to the zip file if successful, or None if the user is not logged in or user history is not initialized.
    """
    # Cannot load history for logged out users
    if profile is None:
        return None
    username = profile if isinstance(profile, str) else profile["preferred_username"]

    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return None

    # Zip history
    with user_history._user_lock(username):
        path = shutil.make_archive(
            str(_archives_path() / f"history_{username}"), "zip", user_history._user_path(username)
        )

    return gr.update(visible=True, value=path)


def _delete_user_history(profile: gr.OAuthProfile | None) -> None:
    """
    Delete all history for the given user.

    Args:
        profile (gr.OAuthProfile | None): The user profile.

    Returns:
        None
    """
    # Cannot load history for logged out users
    if profile is None:
        return
    username = profile if isinstance(profile, str) else profile["preferred_username"]

    user_history = _UserHistory()
    if not user_history.initialized:
        warnings.warn("User history is not set in Gradio demo. You must use `user_history.render(...)` first.")
        return

    with user_history._user_lock(username):
        shutil.rmtree(user_history._user_path(username))


####################
# Internal helpers #
####################


def _copy_image(image: Image.Image | np.ndarray | str | Path, dst_folder: Path, uniqueId: str = "") -> Path:
    try:
        """Copy image to the images folder."""
        # If image is a string, check if it's a URL.
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                return download_and_save_image(image, dst_folder)
            else:
                # Assume it's a local filepath string.
                image = Path(image)

        if isinstance(image, Path):
            dst = dst_folder / Path(f"{uniqueId}_{Path(image).name}")  # keep file ext
            shutil.copyfile(image, dst)
            return dst

        # Still a Python object => serialize it
        if isinstance(image, np.ndarray):
            image = Image.Image.fromarray(image)
        if isinstance(image, Image):
            dst = dst_folder / Path(f"{Path(image).name}")
            image.save(dst)
            return dst

        raise ValueError(f"Unsupported image type: {type(image)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        if not isinstance(dst, Path):
            dst = Path(image)
        return dst  # Return the original file_location if an error occurs

def _copy_file(file: Any | np.ndarray | str | Path, dst_folder: Path, uniqueId: str = "") -> Path:
    try:
        """Copy file to the appropriate folder."""
        # Already a path => copy it
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            dst = dst_folder / Path(f"{file.stem}_{uniqueId}{file.suffix}")  # keep file ext
            shutil.copyfile(file, dst)
            return dst

        # Still a Python object => serialize it
        if isinstance(file, np.ndarray):
            file = Image.fromarray(file)
            dst = dst_folder / Path(f"{file.stem}_{uniqueId}{file.suffix}")
            file.save(dst)
            return dst

        # try other file types
        kind = filetype.guess(file)
        if kind is not None:
            dst = dst_folder / Path(f"{Path(file).stem}_{uniqueId}.{kind.extension}")
            shutil.copyfile(file, dst)
            return dst
        raise ValueError(f"Unsupported file type: {type(file)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        if not isinstance(dst, Path):
            dst = Path(file)
        return dst  # Return the original file_location if an error occurs


def _add_metadata(file_location: Path, metadata: Dict[str, Any], support_path: str = None) -> Path:
    try:
        file_type = file_location.suffix
        valid_file_types = [".wav", ".mp3", ".mp4", ".png"]
        if file_type not in valid_file_types:
            raise ValueError("Invalid file type. Valid file types are .wav, .mp3, .mp4, .png")

        directory, filename, name, ext, new_ext = _get_file_parts(file_location)
        new_file_location = _rename_file_to_lowercase_extension(os.path.join(directory, name +"_h"+ new_ext))

        if file_type == ".wav":
            # Open and process .wav file
            with wave.open(str(file_location), 'rb') as wav_file:
                # Get the current metadata
                current_metadata = {key: value for key, value in wav_file.getparams()._asdict().items() if isinstance(value, (int, float))}
                
                # Update metadata
                current_metadata.update(metadata)

            # Copy the WAV file
                with wave.open(new_file_location, 'wb') as wav_output_file:
                    wav_output_file.setparams(wav_file.getparams())
                    wav_output_file.writeframes(wav_file.readframes(wav_file.getnframes()))
            return new_file_location
        elif file_type == ".mp3":
            # Open and process .mp3 file
            audio = EasyMP3(file_location)

            # Add metadata to the file
            for key, value in metadata.items():
                audio[key] = value

            # Save the MP3 file to the new file location
            audio.save(new_file_location)
            return new_file_location
        elif file_type == ".mp4":
            # Open and process .mp4 file
            # Add metadata to the file
            wav_file_location = Path(support_path) if support_path is not None else file_location.with_suffix(".wav")
            wave_exists = wav_file_location.exists()
            if not wave_exists:
                # Use torchaudio to create the WAV file if it doesn't exist
                audio, sample_rate = torchaudio.load(str(file_location), normalize=True)
                torchaudio.save(wav_file_location, audio, sample_rate, format='wav')

            # Use ffmpeg to add metadata to the video file
            metadata_args = [f"{key}={value}" for key, value in metadata.items()]
            ffmpeg_metadata = ":".join(metadata_args)
            ffmpeg_cmd = f'ffmpeg -y -i "{str(file_location)}" -i "{str(wav_file_location)}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -metadata "{ffmpeg_metadata}" "{new_file_location}"'
            subprocess.run(ffmpeg_cmd, shell=True, check=True)

            # Remove temporary WAV file
            if not wave_exists:
                wav_file_location.unlink()
            return new_file_location
        elif file_type == ".png":
            image = Image.open(str(file_location))
            # Create a PngInfo object for custom metadata
            pnginfo = PngImagePlugin.PngInfo()
            
            for key, value in metadata.items():
                pnginfo.add_text(str(key), str(value))
            
            image.save(str(new_file_location), pnginfo=pnginfo)
            return new_file_location

        return file_location  # Return the path to the modified file

    except Exception as e:
        print(f"An error occurred: {e}")
        return file_location  # Return the original file_location if an error occurs
   
def _resolve_folder_path(folder_path: str | Path | None) -> Path:
    if folder_path is not None:
        return Path(folder_path).expanduser().resolve()

    if os.getenv("SYSTEM") == "spaces" and os.path.exists("/data"):  # Persistent storage is enabled!
        return Path("/data") / "_user_history"

    # Not in a Space or Persistent storage not enabled => local folder
    return Path("_user_history").resolve()


def _archives_path() -> Path:
    # Doesn't have to be on persistent storage as it's only used for download
    path = Path(__file__).parent / "_user_history_exports"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_file_parts(file_path: str):
    # Split the path into directory and filename
    directory, filename = os.path.split(file_path)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Convert the extension to lowercase
    new_ext = ext.lower()
    return directory, filename, name, ext, new_ext

def _rename_file_to_lowercase_extension(file_path: str) -> str:
    """
    Renames a file's extension to lowercase in place.

    Parameters:
        file_path (str): The original file path.

    Returns:
        str: The new file path with the lowercase extension.

    Raises:
        OSError: If there is an error renaming the file (e.g., file not found, permissions issue).
    """
    directory, filename, name, ext, new_ext = _get_file_parts(file_path)
    # If the extension changes, rename the file
    if ext != new_ext:
        new_filename = name + new_ext
        new_file_path = os.path.join(directory, new_filename)
        try:
            os.rename(file_path, new_file_path)
            print(f"Rename {file_path} to {new_file_path}\n")
        except Exception as e:
            print(f"os.rename failed: {e}. Falling back to binary copy operation.")
            try:
                # Read the file in binary mode and write it to new_file_path
                with open(file_path, 'rb') as f:
                    data = f.read()
                with open(new_file_path, 'wb') as f:
                    f.write(data)
                    print(f"Copied {file_path} to {new_file_path}\n")
                # Optionally, remove the original file after copying
                #os.remove(file_path)
            except Exception as inner_e:
                print(f"Failed to copy file from {file_path} to {new_file_path}: {inner_e}")
                raise inner_e
        return new_file_path
    else:
        return file_path

def _get_unique_file_path(directory, filename, file_ext, counter=0):
    """
    Recursively increments the filename until a unique path is found.
    
    Parameters:
        directory (str): The directory for the file.
        filename (str): The base filename.
        file_ext (str): The file extension including the leading dot.
        counter (int): The current counter value to append.
        
    Returns:
        str: A unique file path that does not exist.
    """
    if counter == 0:
        filepath = os.path.join(directory, f"{filename}{file_ext}")
    else:
        filepath = os.path.join(directory, f"{filename}{counter}{file_ext}")

    if not os.path.exists(filepath):
        return filepath
    else:
        return _get_unique_file_path(directory, filename, file_ext, counter + 1)

def _download_and_save_image(url: str, dst_folder: Path) -> Path:
    """
    Downloads an image from a URL, verifies it with PIL, and saves it in dst_folder with a unique filename.
    
    Args:
        url (str): The image URL.
        dst_folder (Path): The destination folder for the image.
        
    Returns:
        Path: The saved image's file path.
    """
    response = requests.get(url)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content))
    
    parsed_url = urlparse(url)
    original_filename = os.path.basename(parsed_url.path)  # e.g., "background.png"
    base, ext = os.path.splitext(original_filename)
    
    # Use get_unique_file_path from file_utils.py to generate a unique file path.
    unique_filepath_str = _get_unique_file_path(str(dst_folder), base, ext)
    dst = Path(unique_filepath_str)
    pil_image.save(dst)
    return dst

#################
# Admin section #
#################


def _admin_section() -> None:
    title = gr.Markdown()
    title.attach_load_event(_display_if_admin(), every=None)


def _display_if_admin() -> Callable:
    def _inner(profile: gr.OAuthProfile | None) -> str:
        if profile is None:
            return ""
        if profile["preferred_username"] in _fetch_admins():
            return _admin_content()
        if profile in _fetch_admins():
            return _admin_content()
        return ""

    return _inner


def _admin_content() -> str:
    return f"""
## Admin section

Running on **{os.getenv("SYSTEM", "local")}** (id: {os.getenv("SPACE_ID")}). {_get_msg_is_persistent_storage_enabled()}

Admins: {', '.join(_fetch_admins())}

{_get_nb_users()} user(s), {_get_nb_images()} image(s), {_get_nb_video()} video(s) in history.

Display Type: *{_UserHistory().display_type}*

### Configuration

History folder: *{_UserHistory().folder_path}*

Exports folder: *{_archives_path()}*

### Disk usage

{_disk_space_warning_message()}
"""


def _get_nb_users() -> int:
    user_history = _UserHistory()
    if not user_history.initialized:
        return 0
    if user_history.folder_path is not None and user_history.folder_path.exists():
        return len([path for path in user_history.folder_path.iterdir() if path.is_dir()])
    return 0


def _get_nb_images() -> int:
    user_history = _UserHistory()
    if not user_history.initialized:
        return 0
    if user_history.folder_path is not None and user_history.folder_path.exists():
        return len([path for path in user_history.folder_path.glob("*/images/*")])
    return 0

def _get_nb_video() -> int:
    user_history = _UserHistory()
    if not user_history.initialized:
        return 0
    if user_history.folder_path is not None and user_history.folder_path.exists():
        return len([path for path in user_history.folder_path.glob("*/videos/*")])
    return 0


def _get_msg_is_persistent_storage_enabled() -> str:
    if os.getenv("SYSTEM") == "spaces":
        if os.path.exists("/data"):
            return "Persistent storage is enabled."
        else:
            return (
                "Persistent storage is not enabled. This means that user histories will be deleted when the Space is"
                " restarted. Consider adding a Persistent Storage in your Space settings."
            )
    return ""


def _disk_space_warning_message() -> str:
    user_history = _UserHistory()
    if not user_history.initialized:
        return ""

    message = ""
    if user_history.folder_path is not None:
        total, used, _ = _get_disk_usage(user_history.folder_path)
        message += f"History folder: **{used / 1e9 :.0f}/{total / 1e9 :.0f}GB** used ({100*used/total :.0f}%)."

    total, used, _ = _get_disk_usage(_archives_path())
    message += f"\n\nExports folder: **{used / 1e9 :.0f}/{total / 1e9 :.0f}GB** used ({100*used/total :.0f}%)."

    return f"{message.strip()}"


def _get_disk_usage(path: Path) -> Tuple[int, int, int]:
    for path in [path] + list(path.parents):  # first check target_dir, then each parents one by one
        try:
            return shutil.disk_usage(path)
        except OSError:  # if doesn't exist or can't read => fail silently and try parent one
            pass
    return 0, 0, 0


@cache
def _fetch_admins() -> List[str]:
    # Running locally => fake user is admin
    if os.getenv("SYSTEM") != "spaces":
        return ["FakeGradioUser"]

    # Running in Space but no space_id => ???
    space_id = os.getenv("SPACE_ID")
    if space_id is None:
        return ["Unknown"]

    # Running in Space => try to fetch organization members
    # Otherwise, it's not an organization => namespace is the user
    namespace = space_id.split("/")[0]
    response = requests.get(f"https://huggingface.co/api/organizations/{namespace}/members")
    if response.status_code == 200:
        return sorted((member["user"] for member in response.json()), key=lambda x: x.lower())
    return [namespace]
