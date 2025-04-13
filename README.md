---
title: Gradio User History
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
emoji: 🖼️
colorFrom: gray
colorTo: indigo
pinned: false
fullWidth: true
hf_oauth: true
space_ci:
  trusted_authors: 
  private: auto
  secrets: 
    - SPACE_CI_SECRET
    - HF_TOKEN
  hardware: cpu-basic
---

# Bring User History to your Spaces 🚀

**Gradio User History** is a plugin (and package) that caches generated images for your Space users.

## Key features:

- 🤗 **Sign in with Hugging Face**
- **Save** generated image, video, audio and document files with their metadata: prompts, timestamp, hyper-parameters, etc.
- **Export** your history as zip.
- **Delete** your history to respect privacy.
- Compatible with **Persistent Storage** for long-term storage.
- **Admin** panel to check configuration and disk usage .
- **Progress bar** to show status of saving files
- Choose between images or video **Gallery**
- Compatible with **Gradio 4.0+** and **Gradio 5.0+**

Want more? Please open an issue in the [Community Tab](https://huggingface.co/spaces/Wauplin/gradio-user-history/discussions)! This is meant to be a community-driven implementation, enhanced by user feedback and contributions!

## Integration

Integrate *Gradio User History* in just a few steps:

**1. Enable OAuth to your Space**

Simply edit your Space metadata in your `README.md` file:


```yaml
# README.md
hf_oauth: true
```


**2. Update your `requirements.txt`**


```bash
# requirements.txt
git+https://huggingface.co/spaces/Surn/gradio-user-history
filetype @ git+https://github.com/h2non/filetype.py.git
gradio[oauth]
mutagen
tqdm
torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124
torchaudio>=2.0.0,<2.6.2 --extra-index-url https://download.pytorch.org/whl/cu124
```


**3. Integrate into your Gradio app**


```py
# app.py
import gradio as gr
import gradio_user_history as gr_user_history
(...)

# => Inject gr.OAuthProfile
def generate(prompt: str, profile: gr.OAuthProfile | None):
    image = ...

    # => Save generated image(s)
    gr_user_history.save_image(label=prompt, image=image, profile=profile)

    # => Save generated image, video, audio, document, metadata
    gr_user_history.save_file(profile=profile,image=image, video=video, audio=audio, document=document,label=string, metadata=metadata)


# => Render user history
with gr.Blocks() as demo:
    (...)

    with gr.Accordion("Past generations", open=False):        
        # => OPTIONALLY display images or videos in the history gallery with display_type: "image_path" or "video_path"
        gr_user_history.setup(display_type="image_path") 
        gr_user_history.render()
```

**4. (optional) Add Persistent Storage in your Space settings.**
   Persistent Storage is suggested but not mandatory. If not enabled, the history is lost each time the Space restarts.

And you're done!

## Useful links

- **Demo:** https://huggingface.co/spaces/Wauplin/gradio-user-history
- **README:** https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/README.md
- **Source file:** https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/src/gradio_user_history/_user_history.py
- **Questions and feedback:** https://huggingface.co/spaces/Wauplin/gradio-user-history/discussions
- **Development Updates**: https://huggingface.co/spaces/Surn/gradio-user-history/

## Preview

![Image preview](https://huggingface.co/spaces/Surn/gradio-user-history/resolve/main/assets/screenshot.png)