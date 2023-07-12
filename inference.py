import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.distributed as dist
from PIL import Image
from project.models.model import MappingType, ClipCaptionModel
id = 391895
from transformers import GPT2Tokenizer
from pycocotools.coco import COCO
import torch.utils.data as data
from clip1.clip import _transform
import json
import skimage.io as io1
from utils.misc import generate2_adpt_if_nodist, evaluate_on_coco_caption

dist.init_process_group("nccl", init_method='file:///tmp/somefile', rank=0, world_size=1)

# id = 209868
# # class DataLoader(data.Dataset):
# #     def __init__(self, json,  transform=None):
# #         self.coco = COCO(json)
# #         self.ids = list(self.coco.anns.keys())
# #         self.transform = transform

# base_path = '../dataset/MSCOCO_Caption/annotations/captions_val2014.json'
# with open(base_path,'r') as f:
#     dataset=json.load(f)
# for data in dataset['annotations']:
#     if data['image_id'] == id:
#         cap = data['caption']
#         break

# raw_image = Image.open('../dataset/MSCOCO_Caption/val2014/COCO_val2014_000000'+str(id)+'.jpg').convert("RGB")
# raw_image.show(title=cap)

image = io1.imread('../dataset/MSCOCO_Caption/val2014/COCO_val2014_000000209868.jpg')
# image = io1.imread('./images/walk.jpg')
image = Image.fromarray(image)
# image.show(title=cap)
image = _transform(224)(image)

model = ClipCaptionModel(10, clip_length=10, prefix_size=512,
                                 num_layers=8, mapping_type=MappingType.MLP, Timestep=20,
                                 if_drop_rate=0.1)


image = image.cuda(non_blocking=True)
kw = 'toothbrush'
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
kt = torch.tensor(tokenizer.encode(kw))
padding = 5 - kt.shape[0]
if padding > 0:
    kt = torch.cat((kt, torch.zeros(padding, dtype=torch.int64) - 1))
elif padding < 0:
    kt = kt[:5]
mask_kt = kt.ge(0)
kt[~mask_kt] = 0
kt = kt.cuda(non_blocking=True)

model.load_state_dict(torch.load('./checkpoints/keyword_full_run_42/keyword_full_run_42-best.pt', map_location=torch.device('cpu'))["model"])
model = model.cuda()
model.eval()
image = image.unsqueeze(0)
kt = kt.unsqueeze(0)
prefix, len_cls = model.image_encode(image)
prefix_embed = model.clip_project(prefix)
kt = model.gpt.transformer.wte(kt)
len_pre = model.len_head(len_cls)
prefix_embed = model.kw_att(prefix_embed, kt)
generated_text_prefix = generate2_adpt_if_nodist(model, tokenizer, embed=prefix_embed, len_pre=len_pre.argmax(-1) + 1)

print(generated_text_prefix)

