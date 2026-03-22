import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np

def tensor2img(tensor):
    to_pil = T.ToPILImage()
    tensor = tensor.to(dtype=torch.float32).cpu()
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    img = to_pil(tensor).convert("RGB")
    return img

def clip_poison_image(clean_path, poison_path, diff_range=12/255):
    max_diff = int(diff_range * 127.5)

    clean_img = Image.open(clean_path).convert('RGB')
    poison_img = Image.open(poison_path).convert('RGB')

    clean_np = np.array(clean_img).astype(np.int16)
    poison_np = np.array(poison_img).astype(np.int16)

    diff = poison_np - clean_np
    diff = np.clip(diff, -max_diff, max_diff)

    clipped_np = np.clip(clean_np + diff, 0, 255).astype(np.uint8)

    clipped_img = Image.fromarray(clipped_np)
    clipped_img.save(poison_path)
