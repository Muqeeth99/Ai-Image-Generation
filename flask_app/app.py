from flask import Flask, request, render_template
import io
import torch
import  gc
# from torch import autocast
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline

app = Flask(__name__, template_folder='template')


assert torch.cuda.is_available()
torch.cuda.empty_cache()
gc.collect()

pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token="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",
).to("cuda")

def run_inference(prompt):
    # with autocast("cuda"):
    with torch.cuda.amp.autocast(True):
        image = pipe(prompt)["sample"][0]  
    img_data = io.BytesIO()
    image.save(img_data, "PNG")
    img_data.seek(0)
    return img_data

@app.route("/")
def myapp():
    return render_template("index.html", title="AI txt2img")


@app.route('/results', methods= ["POST"])
def submit():
    if request.method == 'POST':
        prompt = request.form.get("Prompt")
        img_data = run_inference(prompt)
    return render_template("results.html",n=img_data)