---
title: Gradio User History
emoji: 🖼️
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 3.44.4
app_file: app.py
pinned: false
hf_oauth: true
---

# Bring User History to your Spaces 🚀

***User History*** is a plugin that you can add to your Spaces to cache generated images for your users.

## Key features:

- 🤗 **Sign in with Hugging Face**
- **Save** generated images with their metadata: prompts, timestamp, hyper-parameters, etc.
- **Export** your history as zip.
- **Delete** your history to respect privacy.
- Compatible with **Persistent Storage** for long-term storage.
- **Admin** panel to check configuration and disk usage .

Want more? Please open an issue in the [Community Tab](https://huggingface.co/spaces/Wauplin/gradio-user-history/discussions)! This is meant to be a community-driven implementation, enhanced by user feedback and contributions!

## Integration

To integrate ***User History***, only a few steps are required:
1. Enable OAuth in your Space by adding `oauth: true` to your README (see [here](https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/README.md?code=true#L10))
2. Add a Persistent Storage in your Space settings. Without it, the history will not be saved permanently. Every restart of your Space will erase all the data. If you start with a small tier, it's always possible to increase the disk space later without loosing the data.
3. Copy [`user_history.py`](https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/user_history.py) at the root of your project.
4. Import in your main file with `import user_history` (see [here](https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/app.py#L10))
5. Integrate to your `generate`-like methods. Any function called by Gradio and that generates one or multiple images is a good candidate.
   1. Add `profile: gr.OAuthProfile | None` as argument to the function (see [here](https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/app.py#L16)). This will tell Gradio that it needs to inject the user profile for you.
   2. Use `user_history.save_image(label=..., image=..., profile=profile, metadata=...)` (as done [here](https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/app.py#L32))
      1. `label` is the label of the image. Usually the prompt used to generate it.
      2. `image` is the generated image. It can be a path to a stored image, a `PIL.Image` object or a numpy array.
      3. `profile` is the user profile injected by Gradio
      4. `metadata` (optional) is any additional information you want to add. It has to be a json-able dictionary.
   3. Finally use `user_history.render()` to render the "Past generations" section (see [here](https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/app.py#L53)). A good practice is to set it in a different tab to avoid overloading your first page. You don't have to modify anything of your existing `gr.Blocks` section: just render it inside a Tab.

## Example

Here is a minimal example illustrating what we saw above.

```py
import gradio as gr
import user_history  # 0. Import user_history

# 1. Inject user profile
def generate(prompt: str, profile: gr.OAuthProfile | None):
    image = ...

    # 2. Save image
    user_history.save_image(label=prompt, image=image, profile=profile)
    return image


with gr.Blocks(css="style.css") as demo:
    with gr.Group():
        prompt = gr.Text(show_label=False, placeholder="Prompt")
        gallery = gr.Image()
    prompt.submit(fn=generate, inputs=prompt, outputs=gallery)

# 3. Render user history
with gr.Blocks() as demo_with_history:
    with gr.Tab("Demo"):
        demo.render()
    with gr.Tab("Past generations"):
        user_history.render()

demo_with_history.queue().launch()
```

## Useful links

- **Demo:** https://huggingface.co/spaces/Wauplin/gradio-user-history
- **README:** https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/README.md
- **Source file:** https://huggingface.co/spaces/Wauplin/gradio-user-history/blob/main/user_history.py
- **Discussions:** https://huggingface.co/spaces/Wauplin/gradio-user-history/discussions

## Preview

![Image preview](https://huggingface.co/spaces/Wauplin/gradio-user-history/resolve/main/assets/screenshot.png)