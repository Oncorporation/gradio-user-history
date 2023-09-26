#!/usr/bin/env python

import json
import pathlib
import tempfile

import gradio as gr
from gradio_client import Client

import user_history


client = Client("runwayml/stable-diffusion-v1-5")


def generate(prompt: str, profile: gr.OAuthProfile | None) -> tuple[str, list[str]]:
    out_dir = client.predict(prompt, fn_index=1)

    metadata = {
        "prompt": prompt,
        "negative_prompt": "",
        "guidance_scale": 0.9,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as metadata_file:
        json.dump(metadata, metadata_file)

    with (pathlib.Path(out_dir) / "captions.json").open() as f:
        paths = list(json.load(f).keys())

    # Saving user history
    for path in paths:
        user_history.save_image(label=prompt, image=path, profile=profile, metadata=metadata)

    return paths  # type: ignore


with gr.Blocks(css="style.css") as demo:
    with gr.Group():
        prompt = gr.Text(show_label=False, placeholder="Prompt")
        gallery = gr.Gallery(
            show_label=False,
            columns=2,
            rows=2,
            height="600px",
            object_fit="scale-down",
        )
    prompt.submit(fn=generate, inputs=prompt, outputs=gallery)

with gr.Blocks() as demo_with_history:
    with gr.Tab("App"):
        demo.render()
    with gr.Tab("Past generations"):
        user_history.render()

if __name__ == "__main__":
    demo_with_history.queue().launch()
