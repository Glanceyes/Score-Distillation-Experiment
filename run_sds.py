from PIL import Image
from glob import glob
from typing import *
from utils.utils import load_model, load_image
from pipeline_sds import SDSPipeline

import os
import torch
import argparse
import numpy as np


def run_by_img(
    pipeline: SDSPipeline,
    img: Union[torch.FloatTensor, Image.Image, np.ndarray],
    prompt: str,
    num_inference_steps: int = 200,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    save_dir: str = "./outputs",
    save_noises: bool = False,
):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    result = pipeline(
        prompt=prompt,
        img=img,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        save_dir=save_dir,
        save_noises=save_noises,
    )
    
    result.save(os.path.join(save_dir, f"{'_'.join(prompt)}_by_SDS.png"))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None, help=' image path')
    parser.add_argument('--prompt', type=str, help='source prompt')
    parser.add_argument('--n_steps', type=int, default=1000, help="number of inference steps")
    parser.add_argument('--n_train_steps', type=int, default=1000, help="number of train steps")
    parser.add_argument('--save_dir', type=str, default='./outputs', help="directory for saving target output")
    parser.add_argument('--save_noises', action='store_true', default=False, help="save noises through forward process at each step")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="guidance scale")
    parser.add_argument('--min_percent', type=float, default=0., help="min percent")
    parser.add_argument('--max_percent', type=float, default=1., help="max percent")
    parser.add_argument('--loss_weight', type=float, default=1, help="loss weight")
    parser.add_argument('--save_img_steps', type=int, default=50, help="save img steps")
    parser.add_argument('--cuda', type=int, default=0, help="gpu device id")
    parser.add_argument('--torch_dtype', type=str, default="no", choices=["no", "fp16", "bf16"], help="dtype for less vram memory")
    parser.add_argument('--v2_1', action='store_true', default=False, help="use stable diffusion v2.1")
   
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = load_model(args, pipeline_type='sds').to(device)
    generator = torch.Generator(device).manual_seed(args.seed)
    prompt = args.prompt

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.img_path:
        img_path = args.img_path
        img = load_image(img_path)
    else:
        sd_pipeline = load_model(args).to(device)
        result = sd_pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        img = result
        img_path = os.path.join(args.save_dir, f"reference.png")
        img.save(img_path)

    run_by_img(
        pipeline=pipeline,
        img=img,
        prompt=prompt,
        num_inference_steps=args.n_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        save_dir=args.save_dir,
        save_noises=args.save_noises,
    )
    
if __name__ == '__main__':    
    main()