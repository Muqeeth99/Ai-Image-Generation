from flask import Flask, request, render_template, send_file
import io
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

app = Flask(__name__, template_folder='template',static_folder='/Users/mac/Documents/SD/Stable-diffuser-main/Static')

assert torch.cuda.is_available()
gc.collect()
torch.cuda.empty_cache()

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
remove_safety = False
auth_token = "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"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=auth_token)
pipe = pipe.to(device)
pipe.enable_attention_slicing()


def run_inference(prompt):
    gc.collect()
    torch.cuda.empty_cache()
    num_images = 1
    text=prompt
    texts = [ text ] * num_images
    with autocast(device):
        image = pipe(texts, guidance_scale=7.5, num_inference_steps=10).images[0] 
    image.save("./Static/gen_image.jpeg")

    return


@app.route("/")
def myapp():
    return render_template("index.html", title="AI txt2img")


@app.route('/results', methods= ["POST"])
def submit():
    if request.method == 'POST':
        prompt = request.form.get("Prompt")
        img_data = run_inference(prompt)
    return render_template("results.html",n=img_data)

if __name__ == '__main__':
    app.run()
