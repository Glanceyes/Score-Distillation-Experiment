from PIL import Image
from pipeline_dds import DDSPipeline
from pipeline_sds import SDSPipeline
from diffusers.schedulers import DDIMScheduler
from transformers import BlipForConditionalGeneration, BlipProcessor

import torch
import torch.nn.functional as F

def load_image(image_path, h=512, w=512):
    image = Image.open(image_path).convert('RGB').resize((h, w))
    
    return image

def load_model(args, pipeline_type='sds'):
    if args.v2_1:
        sd_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        sd_version = "runwayml/stable-diffusion-v1-5"

    weight_dtype = torch.float32
    if args.torch_dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.torch_dtype == 'bf16':
        weight_dtype = torch.bfloat16
        
    opt = {
        "num_train_timesteps": args.n_train_steps or 1000,
        "min_percent": args.min_percent or 0.,
        "max_percent": args.max_percent or 1.,
        "loss_weight": args.loss_weight or 2000,
        "save_img_steps": args.save_img_steps or 50,
    }
    
    if pipeline_type == 'sds':
        pipeline = SDSPipeline.from_pretrained(
            sd_version, 
            torch_dtype=weight_dtype,
            safety_checker=None,
            opt=opt
        )
    elif pipeline_type == 'dds':
        pipeline = DDSPipeline.from_pretrained(
            sd_version, 
            torch_dtype=weight_dtype,
            safety_checker=None,
            opt=opt
        )
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    return pipeline
