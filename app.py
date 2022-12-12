from flask import Flask, request, render_template, send_file
import io
import torch
import torch
from min_dalle import MinDalle 

app = Flask(__name__, template_folder='template',static_folder='/Users/mac/Documents/SD/Stable-diffuser-main/Static')


model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device='cpu',
    is_mega=False, 
    is_reusable=True
)

def run_inference(prompt):

    image = model.generate_image(
    text=prompt,
    seed=5,
    grid_size=1,
    is_seamless=False,
    temperature=1,
    top_k=256,
    supercondition_factor=32,
    is_verbose=False
    )
    img_data = io.BytesIO()
#     image.save("/Users/mac/Documents/SD/Stable-diffuser-main/Static/gen_image.jpeg", "JPEG")
    image.save("./Static/gen_image.jpeg", "JPEG")
    img_data.seek(0)

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
