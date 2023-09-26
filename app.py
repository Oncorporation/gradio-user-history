#!/usr/bin/env python

import json
import pathlib
import tempfile

import gradio as gr
from gradio_client import Client


client = Client("runwayml/stable-diffusion-v1-5")


def generate(prompt: str) -> tuple[str, list[str]]:
    negative_prompt = ""
    guidance_scale = 9.0
    out_dir = client.predict(prompt, fn_index=1)

    config = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance_scale,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
        json.dump(config, config_file)

    with (pathlib.Path(out_dir) / "captions.json").open() as f:
        paths = list(json.load(f).keys())
    return paths


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

    prompt.submit(
        fn=generate,
        inputs=prompt,
        outputs=gallery,
    )

    with gr.Tab("Past generations"):
        gr.Markdown("building...")

if __name__ == "__main__":
    demo.launch()
