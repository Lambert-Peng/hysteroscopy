#%%
import torch

a = torch.load(r"C:\台大碩士資料\實驗室\hysteroscopy\EndoViT\all_dataset\weights\A63_A0_01.pt")
# %%
from functools import partial
from torch import nn
from timm.models.vision_transformer import VisionTransformer
model_weights_path = "C:\台大碩士資料\實驗室\hysteroscopy\EndoViT\endovit_seg.pth"
ckpt = torch.load(model_weights_path, map_location="cpu")
model_weights = ckpt.get('model', ckpt)

model = VisionTransformer(
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
).eval()

loading = model.load_state_dict(model_weights, strict=False)

# %%
import torch
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torch import nn
from huggingface_hub import snapshot_download
import psutil
import threading
import time
from tqdm import tqdm
import sys

def process_single_image(image_path, input_size=224, dataset_mean=[0.3464, 0.2280, 0.2228], dataset_std=[0.2520, 0.2128, 0.2093]):
    # Define the transformations
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Apply the transformations
    processed_image = transform(image)
    return processed_image

image_paths = sorted(Path('demo_images').glob('*.png')) # TODO replace with image path
images = torch.stack([process_single_image(Path(r"C:\台大碩士資料\實驗室\hysteroscopy\EndoViT\all_dataset\images\A63_A0_01.jpg"))])

# %%
model.forward_features(images).shape
# %%  
print(a)

print(a.shape)
# %%
