from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.utils import deprecate, logging, PIL_INTERPOLATION, BaseOutput
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from diffusers.schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

from torch.nn import functional as F

from typing import *
from PIL import Image
from dataclasses import dataclass
from functools import partial
from jaxtyping import Float
from torch import Tensor

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)

from utils.attention import prepare_unet
from utils.loss import SDSLoss

import os
import torch
import torchvision
import warnings
import numpy as np

logger = logging.get_logger(__name__)

class SDSPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        feature_extractor: CLIPImageProcessor,
        safety_checker: StableDiffusionSafetyChecker=None,
        caption_generator: BlipForConditionalGeneration=None,
        caption_processor: BlipProcessor=None,
        requires_safety_checker: bool = False,
        opt: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            caption_processor=caption_processor,
            caption_generator=caption_generator,
        )
        
        self.opt = opt
        self.num_train_timesteps = self.opt['num_train_timesteps'] if 'num_train_timesteps' in self.opt else 1000
        self.min_percent = self.opt['min_percent'] if 'min_percent' in self.opt else 0.
        self.max_percent = self.opt['max_percent'] if 'max_percent' in self.opt else 1.
        self.loss_weight = self.opt['loss_weight'] if 'loss_weight' in self.opt else 2000
        self.save_img_steps = self.opt['save_img_steps'] if 'save_img_steps' in self.opt else 50
        
        self.lr = self.opt['lr'] if 'lr' in self.opt else 0.1
        self.weight_decay = self.opt['weight_decay'] if 'weight_decay' in self.opt else 0.0
        self.step_size = self.opt['step_size'] if 'step_size' in self.opt else 50
        self.gamma = self.opt['gamma'] if 'gamma' in self.opt else 0.1
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    def prepare_image_latents(self, image, batch_size, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.vae.encode(image).latent_dist.sample(generator)

            latents = self.vae.config.scaling_factor * latents

        if batch_size != latents.shape[0]:
            if batch_size % latents.shape[0] == 0:
                # expand image_latents for batch_size
                deprecation_message = (
                    f"You have passed {batch_size} text prompts (`prompt`), but only {latents.shape[0]} initial"
                    " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                    " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                    " your script to pass as many initial images as text prompts to suppress this warning."
                )
                deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
                additional_latents_per_image = batch_size // latents.shape[0]
                latents = torch.cat([latents] * additional_latents_per_image, dim=0)
            else:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {latents.shape[0]} to {batch_size} text prompts."
                )
        else:
            latents = torch.cat([latents], dim=0)

        return latents

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device,
        num_images_per_prompt,
        do_classifier_free_guidance: Optional[bool] = False,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        # Process prompt if not already embedded
        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            
            
            # Handle sequence truncation, if necessary
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        
        # Duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # Get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # Duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    def perpendicular_component(self, x: Float[Tensor, "B C H W"], y: Float[Tensor, "B C H W"]):
        # get the component of x that is perpendicular to y
        eps = torch.ones_like(x[:, 0, 0, 0]) * 1e-6
        return (
            x
            - (
                torch.mul(x, y).sum(dim=[1, 2, 3])
                / torch.maximum(torch.mul(y, y).sum(dim=[1, 2, 3]), eps)
            ).view(-1, 1, 1, 1)
            * y
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        img: Optional[Union[torch.FloatTensor, Image.Image, np.ndarray]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 200,
        num_iter_per_timestep: int = 1,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        source_z_0: Optional[torch.FloatTensor] = None,
        target_z_T: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        save_dir: Optional[str] = "./outputs",
        save_noises: bool = False,
        use_perpendicular: bool = False,
    ):
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 1. Check inputs. Raise error if not valid
        self.check_inputs(
            prompt, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0


        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )

        null_prompt_embeds = self._encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
        )
 
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)

        alphas = self.scheduler.alphas.to(device)
        alphas_bar = self.scheduler.alphas_cumprod.to(device)

        num_channels_latents = self.unet.config.in_channels

        # 5. Prepare latent variables
        if source_z_0 is None:
            if img is None:
                raise ValueError("You must provide either `source_img` or `source_z_0`.")
            # 3. Preprocess image
            img = self.image_processor.preprocess(img)
            source_z_0 = self.prepare_image_latents(
                img,
                batch_size,
                self.vae.dtype,
                device,
                generator,
            )

        _, noises = self.invert(
            prompt="",
            latents=source_z_0,
            height=height, 
            width=width, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=1.0,
            num_images_per_prompt=num_images_per_prompt,
            cache_noises=True,
            save_noises=save_noises,
            save_dir=save_dir
        )
        
        # 5-1. Prepare target latents
        if target_z_T is not None:
            target_z_T = target_z_T.to(device)
        else:
            target_z_T = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )

        target_z_0, _ = self.ddim_backward(
            latents=target_z_T,
            prompt_embeds=null_prompt_embeds,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=1.0,
            cache_noises=False,
            save_dir=os.path.join(save_dir, 'target'),
            device=device
        )

        target_img_save_dir = os.path.join(save_dir, 'results')
        if not os.path.exists(target_img_save_dir):
            os.makedirs(target_img_save_dir)

        target_img = self.decode_latents(target_z_0).squeeze()
        target_img = Image.fromarray((target_img * 255).astype(np.uint8))
        target_img.save(os.path.join(target_img_save_dir, f'target_{self.num_train_timesteps}.png'))

        self.unet = prepare_unet(self.unet)

        self.min_step = int(self.num_train_timesteps * self.min_percent)
        self.max_step = int(self.num_train_timesteps * self.max_percent)

        sds_loss = SDSLoss(
            device=device,
            unet=self.unet,
            scheduler=self.scheduler
        )

        latents = target_z_0.clone()
        latents.requires_grad = True

        optimizer = torch.optim.SGD(
            [latents], 
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        num_warmup_steps = num_inference_steps - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                if timestep.item() >= self.num_train_timesteps:
                    timestep = torch.tensor(self.num_train_timesteps - 1, device=device)
                t = timestep.item()

                for j in range(num_iter_per_timestep):
                    optimizer.zero_grad()

                    z_t, eps, timestep = sds_loss.noise_input(latents, timestep, noises[t])

                    eps_pred, pred_z0 = sds_loss.get_epsilon_prediction(
                        z_t,
                        timestep=timestep,
                        text_embeddings=prompt_embeds,
                        alpha_bar_t=alphas_bar[t],
                    )
                    
                    w = (1 - alphas_bar[t]).view(-1, 1, 1, 1)

                    if use_perpendicular:
                        grad = w * self.perpendicular_component(eps_pred - eps, eps_pred)
                    else:
                        grad = w * (eps_pred - eps)
                    grad = torch.nan_to_num(grad)

                    with torch.enable_grad():
                        loss = latents * grad.clone()
                        loss = loss.sum() / (latents.shape[0] * latents.shape[1])
                        loss = loss * self.loss_weight
                        
                        logger.info(f"Step {i+1}/{num_inference_steps} - Loss: {loss.item()}")  
                        loss.backward(retain_graph=True)
                    
                    optimizer.step()

                    if i == num_inference_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, timestep, latents)

                if (i+1) % self.save_img_steps == 0:
                    img = self.decode_latents(latents).squeeze()
                    img = Image.fromarray((img * 255).astype(np.uint8))

                    img.save(os.path.join(target_img_save_dir, f'target_{str(t).zfill(3)}.png'))

        result = self.decode_latents(latents).squeeze()
        result = Image.fromarray((result * 255).astype(np.uint8))

        return result


    @torch.no_grad()
    def invert(
        self,
        prompt: Union[str, List[str]],
        img: Union[torch.FloatTensor, Image.Image, np.ndarray] = None,
        latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        forward_iter_stop: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs = None,
        save_dir: Optional[str] = "./outputs",
        verify: bool = False,
        cache_noises: bool = False,
        save_noises: bool = False,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 1. Check inputs. Raise error if not valid
        self.check_inputs(
            prompt, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        if latents is None:
            if img is None:
                raise ValueError("You must provide either `img` or `latents`.")
            latents = self.prepare_image_latents(
                img,
                batch_size,
                self.vae.dtype,
                device,
                generator,
            )
        latents = latents.to(self.device)
        # latents = latents * self.scheduler.init_noise_sigma

        latents, noises = self.ddim_forward(
            latents=latents,
            prompt_embeds=prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            timesteps=timesteps,
            forward_iter_stop=forward_iter_stop,
            save_dir=os.path.join(save_dir, 'forward'),
            device=device,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
            cache_noises=cache_noises,
            save_noises=save_noises
        )
                
        inverted_latents = latents.clone()
                
        if verify:
            logger.info("Verifying the result...")
            self.ddim_backward(
                latents=inverted_latents,
                prompt_embeds=prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                timesteps=timesteps,
                forward_iter_stop=forward_iter_stop,
                save_dir=os.path.join(save_dir, 'verify'),
                device=device,
                callback=callback,
                callback_steps=callback_steps,
                cross_attention_kwargs=cross_attention_kwargs,
                cache_noises=False,
                save_noises=False,
            )

        return inverted_latents, noises
    

    def ddim_forward(
            self,
            latents: torch.FloatTensor,
            prompt_embeds: torch.FloatTensor,
            num_inference_steps: int,
            timesteps: torch.Tensor,
            guidance_scale: float = 1.0,
            cache_noises: bool = False,
            save_noises: bool = False,
            save_dir: Optional[str] = "./outputs",
            forward_iter_stop: Optional[int] = None,
            device: Optional[torch.device] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs = None,
        ):

        noises = {}
        t_steps = timesteps.clone()

        do_classifier_free_guidance = guidance_scale > 1.0

        if cache_noises and save_noises:
            noise_save_dir = os.path.join(save_dir, 'noises')
            if not os.path.exists(noise_save_dir):
                os.makedirs(noise_save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t_steps = reversed(t_steps)

        if forward_iter_stop is None:
            forward_iter_stop = num_inference_steps

        with self.progress_bar(total=forward_iter_stop) as progress_bar:
            for i in range(forward_iter_stop):
                timestep = t_steps[i]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, 
                    timestep,
                    prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if cache_noises:
                    noises[timestep.item()] = noise_pred

                    if save_noises:
                        torch.save(noise_pred, os.path.join(noise_save_dir, f'noise_{str(i).zfill(3)}.pt'))

                        noise_img = self.decode_latents(noise_pred).squeeze()
                        noise_img = Image.fromarray((noise_img * 255).astype(np.uint8))
                        noise_img.save(os.path.join(noise_save_dir, f'noise_{str(i).zfill(3)}.png'))

                t = timestep.item()
                t = min(t, self.num_train_timesteps - 1)
                prev_t = max(0, t - self.num_train_timesteps // num_inference_steps)

                alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
                alpha_bar_t_prev = self.scheduler.alphas_cumprod[prev_t].to(device)

                pred_z_0 = (latents - torch.sqrt(1 - alpha_bar_t_prev) * noise_pred) / torch.sqrt(alpha_bar_t_prev)
                dir_z_t = torch.sqrt(1. - alpha_bar_t) * noise_pred
                latents = alpha_bar_t.sqrt() * pred_z_0 + dir_z_t
                
                if i == num_inference_steps - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, timestep, latents)
                
                if (i+1) % self.save_img_steps == 0:
                    target_img = self.decode_latents(latents).squeeze()
                    target_img = Image.fromarray((target_img * 255).astype(np.uint8))

                    iter_num = str(t).zfill(3)
                    target_img.save(os.path.join(save_dir, f'forward_{iter_num}.png'))
                
                if forward_iter_stop is not None and i == (forward_iter_stop - 1):
                    break

        return latents, noises
    

    def ddim_backward(
            self,
            latents: torch.FloatTensor,
            prompt_embeds: torch.FloatTensor,
            num_inference_steps: int,
            timesteps: torch.Tensor,
            guidance_scale: float = 1.0,
            cache_noises: bool = False,
            save_noises: bool = False,
            save_dir: Optional[str] = "./outputs",
            forward_iter_stop: Optional[int] = None,
            device: Optional[torch.device] = None,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs = None,
        ):

        noises = {}
        t_steps = timesteps.clone()

        do_classifier_free_guidance = guidance_scale > 1.0

        if cache_noises and save_noises:
            noise_save_dir = os.path.join(save_dir, 'noises')
            if not os.path.exists(noise_save_dir):
                os.makedirs(noise_save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if forward_iter_stop is None:
            forward_iter_stop = num_inference_steps

        with self.progress_bar(total=forward_iter_stop) as progress_bar:
            for i in range(num_inference_steps - forward_iter_stop, num_inference_steps):        
                timestep = t_steps[i]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, 
                    timestep,
                    prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if cache_noises:
                    noises[timestep.item()] = noise_pred

                    if save_noises:
                        torch.save(noise_pred, os.path.join(noise_save_dir, f'noise_{str(i).zfill(3)}.pt'))

                        noise_img = self.decode_latents(noise_pred).squeeze()
                        noise_img = Image.fromarray((noise_img * 255).astype(np.uint8))
                        noise_img.save(os.path.join(noise_save_dir, f'noise_{str(i).zfill(3)}.png'))

                t = timestep.item()
                t = min(t, self.num_train_timesteps - 1)
                prev_t = max(1, t - self.num_train_timesteps // num_inference_steps)

                alpha_bar_t = self.scheduler.alphas_cumprod[t].to(device)
                alpha_bar_t_prev = self.scheduler.alphas_cumprod[prev_t].to(device)

                pred_z_0 = (latents - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                dir_z_t = torch.sqrt(1. - alpha_bar_t_prev) * noise_pred
                latents = alpha_bar_t_prev.sqrt() * pred_z_0 + dir_z_t
                
                if i == num_inference_steps - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, timestep, latents)
                
                if (i+1) % self.save_img_steps == 0:
                    target_img = self.decode_latents(latents).squeeze()
                    target_img = Image.fromarray((target_img * 255).astype(np.uint8))

                    iter_num = str(t).zfill(3)
                    target_img.save(os.path.join(save_dir, f'backward_{iter_num}.png'))
                
                if forward_iter_stop is not None and i == (forward_iter_stop - 1):
                    break

        return latents, noises