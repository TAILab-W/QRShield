import os
import re
import torch
import shutil
import argparse
import jsonlines

from PIL import Image
from utils import tensor2img
from transformers import BlipProcessor, BlipForConditionalGeneration

import torchvision.transforms as T

def main(args):
    processor = BlipProcessor.from_pretrained(args.blip_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.blip_dir).to(args.device)
    resize = T.transforms.Resize(args.resolution)
    center_crop = T.transforms.CenterCrop(args.resolution)
    
    for artist in os.listdir(args.dataset_dir):
        for division in ["train", "test"]:
            image_dir = f"{args.dataset_dir}/{artist}/{division}"
            if not os.path.isdir(image_dir):
                continue
                
            if division == "train":
                save_dir = f"{args.preprocessed_dir}/{artist}/clean"
                os.makedirs(save_dir, exist_ok=True)
            else:
                save_dir = f"{args.preprocessed_dir}/{artist}/test"
                os.makedirs(save_dir, exist_ok=True)

            metadata = []
            for image_name in os.listdir(image_dir):
                image_path = f"{image_dir}/{image_name}"
                if not image_path.endswith(".png") and not image_path.endswith(".jpg"):
                    continue
                image = Image.open(image_path)
                image = center_crop(resize(Image.open(image_path)))
                input = processor(image, return_tensors="pt").to(args.device)
                out = model.generate(**input)
                caption = processor.decode(out[0], skip_special_tokens=True)
                caption = f"{caption} by {artist}"
                
                txt_name = re.sub(r'\.(jpg|png)$', '.txt', image_name)
                image_name = re.sub(r'\.jpg', '.png', image_name)
                with open(f"{save_dir}/{txt_name}", "w", encoding="utf-8") as file:
                    file.write(caption)
                image.save(f"{save_dir}/{image_name}")
                metadata.append({'file_name': image_name, 'text': caption})

            with jsonlines.open(f'{save_dir}/metadata.jsonl', 'w') as writer:
                writer.write_all(metadata)
                
        print(f"✅ Finished processing artist: {artist}")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='diffusion attack')
    parser.add_argument('--blip_dir', default="/root/autodl-tmp/pretrained_model/Blip-image-captioning-base", type=str)
    parser.add_argument('--dataset_dir', default="/root/autodl-tmp/dataset/comtemporary", type=str)
    parser.add_argument('--preprocessed_dir', default="/root/autodl-tmp/dataset/surrogate_SD2.1", type=str)
    parser.add_argument('--resolution', default=512, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    args = parser.parse_args()
    main(args)

