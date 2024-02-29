from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DDPMScheduler, DDIMScheduler

from typing import *

import torch

class SDSLoss:
    def noise_input(self, z, timestep: torch.IntTensor, eps=None):
        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep
    
    def get_epsilon_prediction(self, z_t, timestep, text_embeddings, alpha_bar_t, guidance_scale=7.5, cross_attention_kwargs=None):
        sigma_t = torch.sqrt(1 - alpha_bar_t).to(self.device)
        latent_input = torch.cat([z_t] * 2)

        e_t = self.unet(
            latent_input, 
            timestep, 
            text_embeddings,
            cross_attention_kwargs=cross_attention_kwargs
        ).sample

        if self.prediction_type == 'v_prediction':
            e_t = torch.cat([alpha_bar_t] * 2) * e_t + torch.cat([sigma_t] * 2) * latent_input
        
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        pred_z0 = (z_t - sigma_t * e_t) / torch.sqrt(alpha_bar_t).to(self.device)

        return e_t, pred_z0

    def __init__(
            self,  
            device,
            unet: UNet2DConditionModel,
            scheduler: Optional[Union[DDPMScheduler, DDIMScheduler]],  
        ):
        self.unet = unet
        self.scheduler = scheduler
        self.device = device
        self.prediction_type = scheduler.prediction_type