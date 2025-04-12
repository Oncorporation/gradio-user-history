#!/usr/bin/env python
import json
import pathlib
import tempfile
from pathlib import Path

import gradio as gr
import src.gradio_user_history as gr_user_history
from gradio_client import Client
from gradio_space_ci import enable_space_ci


enable_space_ci()


client = Client("multimodalart/stable-cascade")


def generate(prompt: str, negprompt: str, profile: gr.OAuthProfile | None) -> tuple[str, list[str]]:
    generated_img_path = client.predict(
        prompt,	# str  in 'Prompt' Textbox component
        negprompt,	# str  in 'Negative prompt' Textbox component
        0,	# float (numeric value between 0 and 2147483647) in 'Seed' Slider component
        1536,	# float (numeric value between 1024 and 1536) in 'Width' Slider component
        1536,	# float (numeric value between 1024 and 1536) in 'Height' Slider component
        20,	# float (numeric value between 10 and 30) in 'Prior Inference Steps' Slider component
        4,	# float (numeric value between 0 and 20) in 'Prior Guidance Scale' Slider component
        10,	# float (numeric value between 4 and 12) in 'Decoder Inference Steps' Slider component
        0,	# float (numeric value between 0 and 0) in 'Decoder Guidance Scale' Slider component
        1,	# float (numeric value between 1 and 2) in 'Number of Images' Slider component
        api_name="/run"
    )

    metadata = {
        "prompt": prompt,
        "negative_prompt": negprompt,
        "prior_inference_steps": 20,
        "prior_guidance_scale": 4,
        "decoder_inference_steps": 10,
        "decoder_guidance_scale": 0,
        "seed": 0,
        "width": 1024,
        "height": 1024,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as metadata_file:
        json.dump(metadata, metadata_file)

    # Saving user history
    gr_user_history.save_image(label=prompt, image=generated_img_path, profile=profile, metadata=metadata)

    return [generated_img_path]  # type: ignore


with gr.Blocks(css="style.css", theme='Surn/beeuty@==0.5.12') as demo:
    with gr.Group():
        prompt = gr.Text(show_label=False, placeholder="Prompt")
        negprompt = gr.Text(show_label=False, placeholder="Negative Prompt")
        gallery = gr.Gallery(
            show_label=False,
            columns=2,
            rows=2,
            height="600px",
            object_fit="scale-down",
        )
    prompt.submit(fn=generate, inputs=[prompt,negprompt], outputs=gallery)

with gr.Blocks() as demo_with_history:
    with gr.Tab("README"):
        gr.Markdown(Path("README.md").read_text(encoding="utf-8").split("---")[-1])
    with gr.Tab("Demo"):
        demo.render()
    with gr.Tab("Past generations"):
        gr_user_history.render()

if __name__ == "__main__":
    demo_with_history.queue().launch(allowed_paths=["data/_user_history"])
