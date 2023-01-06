from flask import Flask, request, render_template, send_file
import io
import cv2
import torch
from torch import autocast
from PIL import Image
from diffusers import StableDiffusionPipeline

app = Flask(__name__, template_folder='template',static_folder='./Static')

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

super_res = cv2.dnn_superres.DnnSuperResImpl_create()

def image_to_byte_array(image: Image) -> bytes:
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format=image.format)
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


@app.route('/generate/<string:prompt>')
def run_inference(prompt):
    gc.collect()
    torch.cuda.empty_cache()
    num_images = 1
    text=prompt
    texts = [ text ] * num_images
    with autocast(device):
        image = pipe(texts, guidance_scale=7.5, num_inference_steps=10).images[0] 
    image.save("./Static/gen_image.jpeg", "JPEG")
    
    img = cv2.imread('./Static/gen_image.jpeg')
    super_res.readModel('LapSRN_x8.pb')
    super_res.setModel('lapsrn',8)
    lapsrn_image = super_res.upsample(img)
    cv2.imwrite(r'./Static/Lapsen.jpeg',lapsrn_image)

    img = Image.open('./Static/Lapsen.jpeg', mode='r')
    mime = Image.MIME[img.format]
    img = image_to_byte_array(img)
    return send_file(io.BytesIO(img), mimetype=mime)


if __name__ == '__main__':
    app.run()
