import os
import torch
import random
import argparse
from PIL import Image
from utils import tensor2img, clip_poison_image
from poison_core import pgd_with_momentum, latent_anchor_optimize
from diffusers.models import AutoencoderKL
import warnings
import numpy as np
import torchvision.transforms as T

def parse_args():
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--clean_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    # default
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution of training images (e.g., 512 for 512x512)."
    )
    parser.add_argument(
        "--p_round",
        type=int,
        default=300,
        help="Number of optimization steps for poisoning (PGD with momentum)."
    )
    parser.add_argument(
        "--p_mu",
        type=float,
        default=0.9,
        help="Momentum factor used in perceptual poisoning optimization."
    )
    parser.add_argument(
        "--pgd_lr",
        type=float,
        default=0.5 / 255,
        help="Learning rate for PGD-based poisoning."
    )
    parser.add_argument(
        "--delta_range",
        type=float,
        default=12 / 255,
        help="Maximum perturbation range."
    )
    parser.add_argument(
        "--ag_lr",
        type=float,
        default=0.01,
        help="Learning rate for latent anchor optimization."
    )
    parser.add_argument(
        "--ag_round",
        type=int,
        default=50,
        help="Number of optimization steps for anchor generation."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Size of the latent anchor repeated block unit."
    )
    parser.add_argument(
        "--block_padding",
        type=int,
        default=1,
        help="padding of the latent anchor repeated block unit."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=123456,
        help="seed for reproducibility."
    )
    args = parser.parse_args()
    return args

def load_data(clean_dir):
    image_transforms = T.Compose([
        T.Resize(args.resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(args.resolution),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    clean_images, image_names = [], []
    for file_name in os.listdir(clean_dir):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
            image_names.append(file_name)
            clean_image_path = f"{clean_dir}/{file_name}"
            clean_image = image_transforms(Image.open(clean_image_path).convert("RGB"))
            clean_images.append(clean_image)
            
    clean_tensors = torch.stack(clean_images)
    return clean_tensors, image_names

def main(args):    
    warnings.filterwarnings("ignore", category=UserWarning)
    # get models needed
    vae = AutoencoderKL.from_pretrained(args.pretrained_model,subfolder="vae",torch_dtype=torch.float32)
    # get tensors needed
    clean_tensors, image_names = load_data(args.clean_dir)
    # start poisoning
    os.makedirs(args.save_dir, exist_ok=True)
    target_latent = latent_anchor_optimize(args,vae,clean_tensors)
    for j in range(clean_tensors.shape[0]):
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

        adversarial_tensor = pgd_with_momentum(args,vae,clean_tensors[j, :, :, :].unsqueeze(0),target_latent,j)[0]
            
        adv_img = tensor2img(adversarial_tensor)
        adv_img.save(f"{args.save_dir}/{image_names[j]}")
        # Ensures perturbation stays within ±delta_range after rounding errors from image saving
        clip_poison_image(f"{args.clean_dir}/{image_names[j]}", f"{args.save_dir}/{image_names[j]}", args.delta_range)
            
if __name__ == "__main__":
    args = parse_args()
    torch.backends.cudnn.deterministic=True
    main(args)
