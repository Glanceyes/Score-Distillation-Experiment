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
        source_prompt,
        target_prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        source_prompt_embeds: Optional[torch.FloatTensor] = None,
        target_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        if source_prompt is not None and isinstance(source_prompt, str):
            batch_size = 1
        elif source_prompt is not None and isinstance(source_prompt, list):
            batch_size = len(source_prompt)
        else:
            batch_size = source_prompt_embeds.shape[0]

        if source_prompt_embeds is None:
            text_inputs = self.tokenizer(
                source_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(source_prompt, padding="longest", return_tensors="pt").input_ids

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

        if target_prompt_embeds is None:
            trg_text_inputs = self.tokenizer(
                target_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            trg_text_input_ids = trg_text_inputs.input_ids
            trg_untruncated_ids = self.tokenizer(target_prompt, padding="longest", return_tensors="pt").input_ids

            if trg_untruncated_ids.shape[-1] >= trg_text_input_ids.shape[-1] and not torch.equal(
                trg_text_input_ids, trg_untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    trg_untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
            trg_attention_mask = trg_text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
            trg_attention_mask = None

        source_prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        source_prompt_embeds = source_prompt_embeds[0]

        target_prompt_embeds = self.text_encoder(
            trg_text_input_ids.to(device),
            attention_mask=trg_attention_mask,
        )
        target_prompt_embeds = target_prompt_embeds[0]

        source_prompt_embeds = source_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        target_prompt_embeds = target_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = source_prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        source_prompt_embeds = source_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        source_prompt_embeds = source_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(source_prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(source_prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {source_prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = source_prompt_embeds.shape[1]
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
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return source_prompt_embeds, target_prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        source_prompt: Union[str, List[str]],
        target_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 200,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        source_z_0: Optional[torch.FloatTensor] = None,
        target_z_T: Optional[torch.FloatTensor] = None,
        source_img: Optional[Union[torch.FloatTensor, Image.Image, np.ndarray]] = None,
        target_img: Optional[Union[torch.FloatTensor, Image.Image, np.ndarray]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        target_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        fix_source_noise: bool = False,
        save_dir: Optional[str] = "./outputs",
    ):
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 1. Check inputs. Raise error if not valid
        self.check_inputs(
            source_prompt, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.source_prompt = source_prompt
        if source_prompt is not None and isinstance(source_prompt, str):
            batch_size = 1
        elif source_prompt is not None and isinstance(source_prompt, list):
            batch_size = len(source_prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, target_prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            source_prompt=source_prompt,
            target_prompt=target_prompt if target_prompt is not None else source_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            source_prompt_embeds=prompt_embeds,
            target_prompt_embeds=target_prompt_embeds if target_prompt is not None else prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)

        alphas = self.scheduler.alphas.to(device)
        alphas_bar = self.scheduler.alphas_cumprod.to(device)

        num_channels_latents = self.unet.config.in_channels

        # 5. Prepare latent variables
        if source_z_0 is None:
            if source_img is None:
                raise ValueError("You must provide either `source_img` or `source_z_0`.")
            # 3. Preprocess image
            source_img = self.image_processor.preprocess(source_img)
            source_z_0 = self.prepare_image_latents(
                source_img,
                batch_size,
                self.vae.dtype,
                device,
                generator,
            )

        # 5-1. Prepare target latents
        if target_z_T is not None:
            target_z_T = target_z_T.to(device)

            # scale the initial noise by the standard deviation required by the scheduler
            target_z_T = target_z_T * self.scheduler.init_noise_sigma
        else:
            invert = partial(self.invert, 
                            height=height, 
                            width=width, 
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=num_images_per_prompt,
                            save_dir=save_dir
                            )
            if target_img is None:
                logger.warning("No target image or latents provided. Using source latents as target latents.")
                target_z_T = invert(
                    prompt="",
                    img=source_img,
                    latents=source_z_0,
                )
            else:
                target_img = self.image_processor.preprocess(target_img)
                target_z_T = invert(
                    prompt="",
                    img=target_img,
                )

        self.unet = prepare_unet(self.unet)

        self.min_step = int(self.num_train_timesteps * self.min_percent)
        self.max_step = int(self.num_train_timesteps * self.max_percent)

        sds_loss = SDSLoss(
            device=device,
            unet=self.unet,
            scheduler=self.scheduler
        )

        latents = target_z_T.clone()
        latents.requires_grad = True
        
        # optimizer = torch.optim.SGD(
        #     [latents], 
        #     lr=self.lr,
        #     weight_decay=self.weight_decay
        # )

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=self.step_size,
        #     gamma=self.gamma
        # )
        
        eps_fixed = torch.randn_like(source_z_0) if fix_source_noise else None

        num_warmup_steps = num_inference_steps - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                with torch.enable_grad():
                    t = timestep.item()
                    # optimizer.zero_grad()

                    z_t, eps, timestep = sds_loss.noise_input(source_z_0, timestep, eps_fixed)

                    eps_pred, pred_z0 = sds_loss.get_epsilon_prediction(
                        z_t,
                        timestep=timestep,
                        text_embeddings=prompt_embeds,
                        alpha_bar_t=alphas_bar[t],
                    )
                    
                    w = (1 - alphas_bar[t]).view(-1, 1, 1, 1)

                    grad = w * (eps_pred - eps)

                
                    loss = latents * grad.clone()
                    loss = loss.sum() / (latents.shape[0] * latents.shape[1])
                    loss = loss * self.loss_weight
                    
                    logger.info(f"Step {i+1}/{num_inference_steps} - Loss: {loss.item()}")
                    
                    loss.backward(retain_graph=True)
                
                    # optimizer.step()
                    # scheduler.step()
                
                    gradient = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
                    latents = latents - self.lr * gradient

                if i == num_inference_steps - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if (i+1) % self.save_img_steps == 0:
                    target_img = self.decode_latents(latents).squeeze()
                    target_img = Image.fromarray((target_img * 255).astype(np.uint8))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    target_img.save(os.path.join(save_dir, f'{str(i).zfill(3)}.png'))

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
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        iter_stop: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs = None,
        save_dir: Optional[str] = "./outputs",
        verify: bool = True,
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
        self.source_prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, _, negative_prompt_embeds = self._encode_prompt(
            prompt,
            target_prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            source_prompt_embeds=prompt_embeds,
            target_prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds
        )
        
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

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
        latents = latents * self.scheduler.init_noise_sigma
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(reversed(timesteps)):
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

                prev_timestep = (
                    timestep - self.num_train_timesteps // num_inference_steps
                )

                alpha_bar_t = self.scheduler.alphas_cumprod[timestep].to(device)
                alpha_bar_t_prev = (
                    self.scheduler.alphas_cumprod[prev_timestep]
                    if prev_timestep >= 0
                    else self.scheduler.final_alpha_cumprod
                ).to(device)
                
                pred_z_0 = (latents - torch.sqrt(1 - alpha_bar_t_prev) * noise_pred) / torch.sqrt(alpha_bar_t_prev)
                dir_z_t = torch.sqrt(1. - alpha_bar_t) * noise_pred
                latents = alpha_bar_t.sqrt() * pred_z_0 + dir_z_t
                
                if callback is not None and i % callback_steps == 0:
                    callback(i, timestep, latents)
                
                if (i+1) % self.save_img_steps == 0:
                    target_img = self.decode_latents(latents).squeeze()
                    target_img = Image.fromarray((target_img * 255).astype(np.uint8))

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    target_img.save(os.path.join(save_dir, f'forward_{str(i).zfill(3)}.png'))
                
                if iter_stop is not None and i == iter_stop:
                    break
                
        inverted_latents = latents.clone()
                
        if verify:
            logger.info("Verifying the result...")
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, timestep in enumerate(timesteps[num_inference_steps - iter_stop - 1:] if iter_stop is not None else timesteps):
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

                    prev_timestep = (
                        timestep - self.num_train_timesteps // num_inference_steps
                    )

                    alpha_bar_t = self.scheduler.alphas_cumprod[timestep].to(device)
                    alpha_bar_t_prev = (
                        self.scheduler.alphas_cumprod[prev_timestep]
                        if prev_timestep >= 0
                        else self.scheduler.final_alpha_cumprod
                    ).to(device)
                    
                    pred_z_0 = (latents - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                    dir_z_t = torch.sqrt(1. - alpha_bar_t_prev) * noise_pred
                    latents = alpha_bar_t_prev.sqrt() * pred_z_0 + dir_z_t
                    
                    if callback is not None and i % callback_steps == 0:
                        callback(i, timestep, latents)
                    
                    if (i+1) % self.save_img_steps == 0:
                        target_img = self.decode_latents(latents).squeeze()
                        target_img = Image.fromarray((target_img * 255).astype(np.uint8))

                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        target_img.save(os.path.join(save_dir, f'backward_{str(num_inference_steps - i - 1).zfill(3)}.png'))
            
        return inverted_latents