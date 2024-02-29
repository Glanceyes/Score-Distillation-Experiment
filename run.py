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
    source_img: Union[torch.FloatTensor, Image.Image, np.ndarray],
    source_prompt: str,
    source_img_path: str,
    target_img: Union[torch.FloatTensor, Image.Image, np.ndarray] = None,
    target_prompt: str = None,
    target_img_path: str = None,
    num_inference_steps: int = 200,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    fix_source_noise: bool = False,
    save_dir: str = "./outputs",
):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    result = pipeline(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        num_inference_steps= num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        source_img=source_img,
        target_img=target_img,
        fix_source_noise=fix_source_noise,
        save_dir=save_dir
    )
    
    result.save(os.path.join(save_dir, f"{source_img_path.split('/')[-1].split('.')[0]}_to_{target_img_path.split('/')[-1].split('.')[0]}.png"))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default='samples/cat1.png', help='source image path')
    parser.add_argument('--source_prompt', type=str, help='source prompt')
    parser.add_argument('--target_img_path', type=str, default='samples/dog1.png', help='source image path')
    parser.add_argument('--target_prompt', type=str, help='target prompt')
    parser.add_argument('--n_steps', type=int, default=200, help="number of inference steps")
    parser.add_argument('--n_train_steps', type=int, default=1000, help="number of train steps")
    parser.add_argument('--save_dir', type=str, default='./outputs', help="directory for saving target output")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--min_percent', type=float, default=0., help="min percent")
    parser.add_argument('--max_percent', type=float, default=1., help="max percent")
    parser.add_argument('--loss_weight', type=float, default=1, help="loss weight")
    parser.add_argument('--save_img_steps', type=int, default=50, help="save img steps")
    parser.add_argument('--cuda', type=int, default=0, help="gpu device id")
    parser.add_argument('--torch_dtype', type=str, default="no", choices=["no", "fp16", "bf16"], help="dtype for less vram memory")
    parser.add_argument('--fix_source_noise', action='store_true', default=False, help="fix source noise")
    parser.add_argument('--v2_1', action='store_true', default=False, help="use stable diffusion v2.1")
   
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    pipeline = load_model(args).to(device)
    generator = torch.Generator(device).manual_seed(args.seed)
    
    source_img_path = args.source_img_path
    source_img = load_image(source_img_path)
    
    source_prompt = args.source_prompt
    
    target_img, target_prompt = None, None

    if args.target_img_path is not None:
        target_img_path = args.target_img_path
        target_img = load_image(target_img_path)
        target_prompt = args.target_prompt
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    run_by_img(
        pipeline=pipeline,
        source_img=source_img,
        source_prompt=source_prompt,
        target_img=target_img,
        target_prompt=target_prompt,
        source_img_path=source_img_path,
        target_img_path=target_img_path,
        num_inference_steps=args.n_steps,
        generator=generator,
        fix_source_noise=args.fix_source_noise,
        save_dir=args.save_dir,
    )
    
if __name__ == '__main__':    
    main()