#!/usr/bin/env python
import json
import pathlib
import tempfile
from pathlib import Path
import numpy as np 

import gradio as gr
import src.gradio_user_history as gr_user_history
from modules.version_info import versions_html

from gradio_client import Client
#from gradio_space_ci import enable_space_ci


#enable_space_ci()


client = Client("multimodalart/stable-diffusion-3.5-large-turboX")


def generate(prompt: str, negprompt: str, seed: int, randomize_seed: bool, profile: gr.OAuthProfile | None) -> list[str | None]:
    # API call to the new endpoint
    # The result is a tuple, where the first element is a dictionary containing image information
    # and the second element is the seed.    
    
    if randomize_seed:
        actual_seed = np.random.randint(0, 2147483647 + 1) # Use 2147483647 as MAX_SEED, +1 because randint is exclusive for the upper bound
    else:
        actual_seed = seed

    result = client.predict(
        prompt=prompt,  # str  in 'Prompt' Textbox component
        negative_prompt=negprompt,  # str  in 'Negative prompt' Textbox component
        seed=actual_seed,  # float (numeric value between 0 and 2147483647) in 'Seed' Slider component
        randomize_seed=randomize_seed,  # bool in 'Randomize seed' Checkbox component
        width=1024,  # float (numeric value between 1024 and 1536) in 'Width' Slider component
        height=1024,  # float (numeric value between 1024 and 1536) in 'Height' Slider component
        guidance_scale=1.5,  # float (numeric value between 0 and 20) in 'Guidance scale' Slider component
        num_inference_steps=8,  # float (numeric value between 4 and 12) in 'Number of inference steps' Slider component
        api_name="/infer"
    )

    generated_img_path: str | None = result[0] # Extracting the image path safely
    returned_seed = result[1] # Extracting the seed from the result

    metadata = {
        "prompt": prompt,
        "negative_prompt": negprompt,
        "seed": returned_seed, # Using the seed returned by the API
        "randomize_seed": randomize_seed,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 1.5,
        "num_inference_steps": 8,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as metadata_file:
        json.dump(metadata, metadata_file)

    # Saving user history
    # Ensure generated_img_path is not None if save_image expects a valid path
    if generated_img_path:
        gr_user_history.save_image(label=prompt, image=generated_img_path, profile=profile, metadata=metadata)

    return [generated_img_path]


with gr.Blocks(css="style.css") as demo:
    with gr.Group():
        prompt = gr.Text(show_label=False, placeholder="Prompt")
        negprompt = gr.Text(show_label=False, placeholder="Negative Prompt")
        # Add Seed Slider and Randomize Seed Checkbox
        with gr.Row():
            seed_slider = gr.Slider(minimum=0, maximum=2147483647, step=1, label="Seed", value=0, scale=4)
            randomize_checkbox = gr.Checkbox(label="Randomize seed", value=True, scale=1)
        gallery = gr.Gallery(
            show_label=False,
            columns=2,
            rows=2,
            height="600px",
            object_fit="scale-down",
        )
        submit_button = gr.Button("Generate")

    submit_button.click(fn=generate, inputs=[prompt, negprompt, seed_slider, randomize_checkbox], outputs=gallery)
    prompt.submit(fn=generate, inputs=[prompt, negprompt, seed_slider, randomize_checkbox], outputs=gallery)

with gr.Blocks(theme='Surn/beeuty@==0.5.25') as demo_with_history:
    with gr.Tab("README"):
        gr.Markdown(Path("README.md").read_text(encoding="utf-8").split("---")[-1])
    with gr.Tab("Demo"):
        demo.render()
    with gr.Tab("Past generations"):
        gr_user_history.setup(display_type="image_path") # optional, this is where you would set the display type = "video_path" if you want to display videos
        gr_user_history.render()
    with gr.Row("Versions") as versions_row:
        gr.HTML(value=versions_html(), visible=True, elem_id="versions")


if __name__ == "__main__":
    launch_args = {}
    launch_kwargs = {}
    launch_kwargs['allowed_paths'] = ["assets/", "data/_user_history", "/data/_user_history/Surn"]
    launch_kwargs['favicon_path'] = "assets/favicon.ico"
    #launch_kwargs['inbrowser'] = True

    demo_with_history.queue().launch(**launch_kwargs)
