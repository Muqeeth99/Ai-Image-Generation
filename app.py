from flask import Flask, request, send_file, render_template
import io
import torch
import  gc
# from torch import autocast 
from torch.cuda.amp import autocast
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

assert torch.cuda.is_available()
torch.cuda.empty_cache()

# gc.collect()
# pipe = StableDiffusionPipeline.from_pretrained(
#         "CompVis/stable-diffusion-v1-4", 
#         torch_dtype=torch.float16,
#         revision="fp16",
#         use_auth_token="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",
# ).to("cuda")

def run_inference(prompt):
#   with autocast("cuda"):
#       image = pipe(prompt)["sample"][0]  
#   img_data = io.BytesIO()
#   image.save(img_data, "PNG")
#   img_data.seek(0)
    return prompt
#   return img_data

@app.route('/')
def myapp():
    # if "prompt" not in request.args:
    #     return "Please specify a prompt parameter", 400
    # prompt = request.args["prompt"]
    # img_data = run_inference(prompt)
    return render_template("index.html", title="Hello")
    # return send_file(img_data, mimetype='image/png')


@app.route('/results', methods= ["POST"])
def submit():
    if request.method == 'POST':
        if "prompt" not in request.args:
            return "Please specify a prompt parameter", 400
        prompt = request.form.get("Prompt")
        img_data = run_inference(prompt)
    return render_template("results.html",n=img_data)

# if __name__=='__main__':
#     # app.debug = True
#     app.run(host='0.0.0.0',port=5000) #use_reloder=False,