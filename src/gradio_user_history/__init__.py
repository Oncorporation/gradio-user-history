"""
User History is a plugin that you can add to your Spaces to cache generated images for your users.

Key features:
- 🤗 Sign in with Hugging Face
- Save generated image, video, audio and document files with their metadata: prompts, timestamp, hyper-parameters, etc.
- Export your history as zip.
- Delete your history to respect privacy.
- Compatible with Persistent Storage for long-term storage.
- Admin panel to check configuration and disk usage .
- Progress bar to show status of saving files
- Choose between images or video Gallery

Useful links:
- Demo: https://huggingface.co/spaces/Wauplin/gradio-user-history
- README: https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/README.md
- Source file: https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/user_history.py
- Discussions: https://huggingface.co/spaces/Wauplin/gradio-user-history/discussions
- Development Updates: https://huggingface.co/spaces/Surn/gradio-user-history/
"""
from ._user_history import render, save_image, save_file, setup  # noqa: F401


__version__ = "0.3.4"
