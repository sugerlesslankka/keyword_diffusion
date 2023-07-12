import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# raw_image = Image.open("../images/airballoon.jpeg").convert("RGB")
# raw_image = Image.open("../images/cat.jpeg").convert("RGB")
# raw_image = Image.open("../images/penguin.jpeg").convert("RGB")
# raw_image = Image.open("../images/pedestrian.jfif").convert("RGB")
raw_image = Image.open("../images/man_umbrella.jpg").convert("RGB")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text = model.generate({"image":image})
print(text)