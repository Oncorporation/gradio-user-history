import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


def get_file_parts(file_path: str):
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_ext = ext.lower()
    return directory, filename, name, ext, new_ext


def rename_file_to_lowercase_extension(file_path: str) -> str:
    """
    Renames a file's extension to lowercase in place.

    Parameters:
        file_path (str): The original file path.

    Returns:
        str: The new file path with the lowercase extension.

    Raises:
        OSError: If there is an error renaming the file (e.g., file not found, permissions issue).
    """
    directory, filename, name, ext, new_ext = get_file_parts(file_path)
    if ext != new_ext:
        new_filename = name + new_ext
        new_file_path = os.path.join(directory, new_filename)
        try:
            os.rename(file_path, new_file_path)
            print(f"Rename {file_path} to {new_file_path}\n")
        except Exception as e:
            print(f"os.rename failed: {e}. Falling back to binary copy operation.")
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                with open(new_file_path, "wb") as f:
                    f.write(data)
                    print(f"Copied {file_path} to {new_file_path}\n")
            except Exception as inner_e:
                print(f"Failed to copy file from {file_path} to {new_file_path}: {inner_e}")
                raise inner_e
        return new_file_path
    return file_path


def get_filename(file):
    # extract filename from file object
    filename = None
    if file is not None:
        filename = file.name
    return filename


def convert_title_to_filename(title):
    # convert title to filename
    filename = title.lower().replace(" ", "_").replace("/", "_")
    return filename


def get_filename_from_filepath(filepath):
    file_name = os.path.basename(filepath)
    file_base, file_extension = os.path.splitext(file_name)
    return file_base, file_extension


def delete_file(file_path: str) -> None:
    """
    Deletes the specified file.

    Parameters:
        file_path (str): The path to thefile to delete.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If there is an error deleting the file.
    """
    try:
        path = Path(file_path)
        path.unlink()
        print(f"Deleted original file: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")


def get_unique_file_path(directory, filename, file_ext, counter=0):
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
    return get_unique_file_path(directory, filename, file_ext, counter + 1)


def download_and_save_image(url: str, dst_folder: Path, token: str | None = None) -> Path:
    """
    Downloads an image from a URL, verifies it with PIL, and saves it in dst_folder with a unique filename.

    Args:
        url (str): The image URL.
        dst_folder (Path): The destination folder for the image.

    Returns:
        Path: The saved image's file path.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    pil_image = Image.open(BytesIO(response.content))

    parsed_url = urlparse(url)
    original_filename = os.path.basename(parsed_url.path)
    base, ext = os.path.splitext(original_filename)

    unique_filepath_str = get_unique_file_path(str(dst_folder), base, ext)
    dst = Path(unique_filepath_str)
    dst_folder.mkdir(parents=True, exist_ok=True)
    pil_image.save(dst)
    return dst


def download_and_save_file(url: str, dst_folder: Path, token: str = None) -> Path:
    """
    Downloads a binary file (e.g., audio or video) from a URL with authentication if a token is provided,
    and saves it in dst_folder with a unique filename.

    Args:
        url (str): The file URL.
        dst_folder (Path): The destination folder for the file.
        token (str, optional): A valid Bearer token.

    Returns:
        Path: The saved file's path.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    parsed_url = urlparse(url)
    original_filename = os.path.basename(parsed_url.path)
    base, ext = os.path.splitext(original_filename)

    unique_filepath_str = get_unique_file_path(str(dst_folder), base, ext)
    dst = Path(unique_filepath_str)
    dst_folder.mkdir(parents=True, exist_ok=True)

    with open(dst, "wb") as f:
        f.write(response.content)

    return dst
