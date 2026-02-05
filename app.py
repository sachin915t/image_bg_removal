import gradio as gr
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from typing import Union, Tuple
from PIL import Image

# --------------------------------
# 1. Torch precision
# --------------------------------
torch.set_float32_matmul_precision("high")

# --------------------------------
# 2. Device detection
# --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------------
# 3. Load BiRefNet model
# --------------------------------
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True,
)

birefnet = birefnet.to(device)

# 🔑 CPU FIX: force float32
if device == "cpu":
    birefnet = birefnet.float()

birefnet.eval()

# --------------------------------
# 4. Image preprocessing
# --------------------------------
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# --------------------------------
# 5. Core inference
# --------------------------------
def process(image: Image.Image) -> Image.Image:
    image_size = image.size

    input_images = (
        transform_image(image)
        .unsqueeze(0)
        .to(device)
        .float()   # 🔑 CPU safety
    )

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)

    image = image.convert("RGBA")
    image.putalpha(mask)
    return image

# --------------------------------
# 6. Gradio handlers
# --------------------------------
def fn(image: Union[Image.Image, str]) -> Tuple[Image.Image, Image.Image]:
    im = load_img(image, output_type="pil").convert("RGB")
    origin = im.copy()
    processed = process(im)
    return origin, processed

def process_file(f: str) -> str:
    out_path = f.rsplit(".", 1)[0] + ".png"
    im = load_img(f, output_type="pil").convert("RGB")
    transparent = process(im)
    transparent.save(out_path)
    return out_path

# --------------------------------
# 7. UI components
# --------------------------------
slider1 = gr.ImageSlider(label="Before / After", type="pil", format="png")
slider2 = gr.ImageSlider(label="From URL", type="pil", format="png")

image_upload = gr.Image(label="Upload Image", type="pil")
image_file_upload = gr.Image(label="Upload Image File", type="filepath")
url_input = gr.Textbox(label="Paste Image URL")
output_file = gr.File(label="Download PNG")

url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

tab1 = gr.Interface(
    fn,
    inputs=image_upload,
    outputs=slider1,
    api_name="image",
)

tab2 = gr.Interface(
    fn,
    inputs=url_input,
    outputs=slider2,
    examples=[url_example],
    api_name="url",
)

tab3 = gr.Interface(
    process_file,
    inputs=image_file_upload,
    outputs=output_file,
    api_name="file",
)

# --------------------------------
# 8. Launch app
# --------------------------------
demo = gr.TabbedInterface(
    [tab1, tab2, tab3],
    ["Image Upload", "URL Input", "File Output"],
    title="Sachin's Background Removal App",
)

if __name__ == "__main__":
    demo.launch(show_error=True)
