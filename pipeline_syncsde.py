from typing import Any, Callable, Dict, List, Optional, Union
import math
import numpy as np
import torch

from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

from pipeline_flux_rf_inversion import RFInversionFluxPipeline

from diffusers.image_processor import PipelineImageInput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from torchvision import transforms
import torch.nn.functional as F

import os
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

import math

class RFInversionFluxPipelineSDE(RFInversionFluxPipeline):
    # Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion.prepare_unet
    def prepare_transformer(self, attention_store, AttenProcessorType):
        attn_procs = {}
        for name in self.transformer.attn_processors.keys():
            attn_procs[name] = AttenProcessorType(attention_store=attention_store, name=name)
        self.transformer.set_attn_processor(attn_procs)
    
    def sample_brownian_integral(self, sigmas, num_inference_steps, shape):
        iid_noise = torch.randn(num_inference_steps, *shape, device=self.device, dtype=torch.float32)
        brownian_int = torch.zeros(num_inference_steps+1, *shape, device=self.device, dtype=torch.float32)
        # Compute s and delta_t for all steps
        s = (sigmas[1:] + sigmas[:-1]) / 2.0             # shape: [num_steps]
        delta_t = sigmas[:-1] - sigmas[1:]               # shape: [num_steps]
        mult = ((2 * s).sqrt() / (1 - s) ** 1.5) * delta_t.sqrt()  # shape: [num_steps]

        # reshape mult to be broadcastable with iid_noise (which is [num_steps, ...])
        mult = mult.view(-1, *([1] * (iid_noise.ndim - 1)))  # [num_steps, 1, 1, ...]

        # compute increments and reverse cumulative sum
        increments = iid_noise * mult                      # [num_steps, ...]
        brownian_int[:-1] = torch.flip(torch.cumsum(torch.flip(increments, dims=[0]), dim=0), dims=[0])
        return brownian_int, iid_noise


    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        mask: PipelineImageInput,
        starting_index: int,

        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        source_image_prompt: Union[str, List[str]] = None,
        source_image_prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,

        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 1.0,
        source_guidance_scale: float = 1.0,

        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        resample: bool = False,

        true_cfg_scale : float = 0.0,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 0. process mask if provided
        if mask is not None:
            mask_tensor = transforms.ToTensor()(mask.resize((height, width), 2)).unsqueeze(0)  # shape: (1, 1, H, W)
            binary_mask = (mask_tensor > 0.9).float()  # binarize if needed
            latent_height = int(height) // self.vae_scale_factor // 2
            latent_width = int(width) // self.vae_scale_factor // 2
            latent_mask = F.interpolate(binary_mask, size=(latent_height, latent_width), mode="nearest")  # [1, 1, 64, 64]
            latent_mask_flat = latent_mask.view(-1)  # [1, 4096, 1]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        (
            source_prompt_embeds,
            source_pooled_prompt_embeds,
            source_text_ids,
        ) = self.encode_prompt(
            prompt=source_image_prompt,
            prompt_2=source_image_prompt_2,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 2.5 negative prompts
        has_neg_prompt = negative_prompt or negative_prompt_2
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 3. prepare the process
        dtype = self.text_encoder.dtype # should be torch.bfloat16
        image_latents, _, self.latent_dist = self.encode_image(image, height=height, width=width, dtype=dtype)

        num_channels_latents = self.transformer.config.in_channels // 4
        image_latents, latent_image_ids = self.prepare_latents_inversion(
            batch_size, num_channels_latents, height, width, dtype, device, image_latents
        )
        image_latents, latent_image_ids = image_latents.to(torch.float32), latent_image_ids.to(torch.float32)

        latent_mean, latent_std = self.latent_dist.mean, self.latent_dist.std
        latent_mean = (latent_mean - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_std = latent_std * self.vae.config.scaling_factor
        latent_mean, _ = self.prepare_latents_inversion(
            batch_size, num_channels_latents, height, width, torch.float32, device, latent_mean
        )
        latent_std, _ = self.prepare_latents_inversion(
            batch_size, num_channels_latents, height, width, torch.float32, device, latent_std
        )

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )

        timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(image_latents.shape[0])
            source_guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
            source_guidance = source_guidance.expand(image_latents.shape[0])
        else:
            guidance = None

        # 5. Prepare latent variables
        brownian_int, iid_noise = self.sample_brownian_integral(sigmas, num_inference_steps, image_latents.shape)
        eps = 1e-3 if starting_index == 0 else 0e-3
        diff_latents = torch.zeros_like(image_latents)

        # 6. Denoising loop / Controlled Reverse ODE, Algorithm 2 from: https://arxiv.org/pdf/2410.10792
        with self.progress_bar(total=num_inference_steps-starting_index) as progress_bar:
            for i in range(starting_index, len(timesteps)):
                eps = 1e-3 if i == 0 else 0e-3
                t = timesteps[i]

                if self.interrupt:
                    continue
                
                ################################
                # Process Simulation
                ################################
                def get_preds(latents, pooled_prompt_embeds, prompt_embeds, text_ids, guidance, use_neg_prompt=False):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(dtype),
                        timestep=sigmas[i].expand(latents.shape[0]).to(dtype),
                        guidance=guidance.to(dtype),
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids.to(dtype),
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0].to(torch.float32)
                    if use_neg_prompt and do_true_cfg:
                        neg_noise_pred = self.transformer(
                            hidden_states=latents.to(dtype),
                            timestep=sigmas[i].expand(latents.shape[0]).to(dtype),
                            guidance=guidance.to(dtype),
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids.to(dtype),
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0].to(torch.float32)
                        # noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                        neg_g2score = (2*(-neg_noise_pred) - 2*latents/(1-sigmas[i]))
                    v_t = -noise_pred
                    g2score = (2*v_t - 2*latents/(1-sigmas[i]))
                    if use_neg_prompt and do_true_cfg:
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                        g2score =  neg_g2score +  true_cfg_scale * (g2score - neg_g2score)
                        drift = g2score + latents/(1-sigmas[i]+eps)
                    else:
                        drift = g2score + latents/(1-sigmas[i]+eps)
                    return noise_pred, drift, g2score

                if resample:
                    brownian_int, iid_noise = self.sample_brownian_integral(sigmas, num_inference_steps, image_latents.shape)
                source_latents = brownian_int[i] * (1-sigmas[i]) + image_latents * (1-sigmas[i])

                source_pred, source_drift, g2score_source = get_preds(
                    latents=source_latents,
                    pooled_prompt_embeds=source_pooled_prompt_embeds,
                    prompt_embeds=source_prompt_embeds,
                    text_ids=source_text_ids,
                    guidance=source_guidance,
                )

                latents = diff_latents + source_latents
                noise_pred, drift, g2score_backward = get_preds(
                    latents,
                    pooled_prompt_embeds,
                    prompt_embeds,
                    text_ids,
                    guidance,
                    use_neg_prompt=True
                )
                latents_dtype = latents.dtype
                delta_t = (sigmas[i] - sigmas[i+1])
                diffusion_coeff = (2 * sigmas[i] / (1 - sigmas[i]+eps) * delta_t).sqrt()
                diffusion = g2score_source * delta_t + iid_noise[i] * diffusion_coeff
                ################################
                # Process Simulation
                ################################

                diff_latents = diff_latents +  (drift - source_drift) * delta_t
                if resample:
                    source_latents = image_latents * (1-sigmas[i+1]) + sigmas[i+1] * torch.randn_like(image_latents)
                else:
                    source_latents = brownian_int[i+1] * (1-sigmas[i+1]+eps) + image_latents * (1-sigmas[i+1]+eps)
                latents = diff_latents + source_latents

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()
                logger.debug("i = {:d}, timestep = {:.1f}, sigma_i = {:.4f}".format(i, timesteps[i], sigmas[i]))
                logger.debug("\tdrift = {:.2f}, backward latents norm = {:.2f}, source latents norm= {:.2f}".format(drift.detach().cpu().norm(), latents.detach().cpu().norm(), source_latents.detach().cpu().norm()))


        logger.debug("recon_loss = {:.4f}".format(  (latents.detach().cpu() - image_latents.detach().cpu()).norm() ))

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents.to(dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

