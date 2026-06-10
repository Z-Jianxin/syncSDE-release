from typing import Any, Callable

import numpy as np
import torch
from diffusers import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class Flux2SyncSDEPipeline(Flux2Pipeline):
    def sample_brownian_integral(self, sigmas, num_inference_steps, shape):
        device = sigmas.device if torch.is_tensor(sigmas) else self._execution_device
        sigmas = torch.as_tensor(sigmas, device=device, dtype=torch.float32)
        iid_noise = torch.randn(num_inference_steps, *shape, device=device, dtype=torch.float32)
        brownian_int = torch.zeros(num_inference_steps + 1, *shape, device=device, dtype=torch.float32)

        s = (sigmas[1:] + sigmas[:-1]) / 2.0
        delta_t = sigmas[:-1] - sigmas[1:]
        mult = ((2 * s).sqrt() / (1 - s) ** 1.5) * delta_t.sqrt()
        mult = mult.view(-1, *([1] * (iid_noise.ndim - 1)))

        increments = iid_noise * mult
        brownian_int[:-1] = torch.flip(torch.cumsum(torch.flip(increments, dims=[0]), dim=0), dims=[0])
        return brownian_int, iid_noise

    def get_timesteps(self, num_inference_steps, strength=1.0):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        sigmas = self.scheduler.sigmas[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, sigmas, num_inference_steps - t_start

    def encode_image_latents(self, image, height, width, generator, device, dtype):
        self.image_processor.check_image_input(image)
        image = self.image_processor.preprocess(image, height=height, width=width, resize_mode="crop")
        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        latent_ids = self._prepare_latent_ids(image_latents).to(device)
        image_latents = self._pack_latents(image_latents).to(device=device, dtype=torch.float32)
        return image_latents, latent_ids

    @torch.no_grad()
    def __call__(
        self,
        image,
        starting_index: int,
        prompt: str | list[str] = None,
        source_image_prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        strength: float = 1.0,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float | None = 4.0,
        source_guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (10, 20, 30),
        resample: bool = False,
    ):
        if source_image_prompt is None:
            raise ValueError("`source_image_prompt` is required for FLUX.2 syncSDE editing.")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        multiple_of = self.vae_scale_factor * 2
        height = (int(height) // multiple_of) * multiple_of
        width = (int(width) // multiple_of) * multiple_of

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        guidance_scale = 4.0 if guidance_scale is None else guidance_scale
        source_guidance_scale = guidance_scale if source_guidance_scale is None else source_guidance_scale

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        source_prompt_embeds, source_text_ids = self.encode_prompt(
            prompt=source_image_prompt,
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        self.maybe_free_model_hooks()
        image_latents, latent_ids = self.encode_image_latents(
            image=image,
            height=height,
            width=width,
            generator=generator,
            device=device,
            dtype=self.vae.dtype,
        )
        self.maybe_free_model_hooks()

        effective_batch = batch_size * num_images_per_prompt
        if image_latents.shape[0] != effective_batch:
            image_latents = image_latents.repeat(effective_batch, 1, 1)
            latent_ids = latent_ids.repeat(effective_batch, 1, 1)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=image_latents.shape[1], num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps=timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        sigmas = sigmas.to(device=image_latents.device, dtype=torch.float32)
        if starting_index < 0 or starting_index >= num_inference_steps:
            raise ValueError(f"`starting_index` must be in [0, {num_inference_steps - 1}], got {starting_index}.")

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        guidance = torch.full([1], float(guidance_scale), device=device, dtype=torch.float32)
        guidance = guidance.expand(image_latents.shape[0])
        source_guidance = torch.full([1], float(source_guidance_scale), device=device, dtype=torch.float32)
        source_guidance = source_guidance.expand(image_latents.shape[0])

        brownian_int, iid_noise = self.sample_brownian_integral(sigmas, num_inference_steps, image_latents.shape)
        diff_latents = torch.zeros_like(image_latents)

        with self.progress_bar(total=num_inference_steps - starting_index) as progress_bar:
            for i in range(starting_index, len(timesteps)):
                eps = 1e-3 if i == 0 else 0e-3
                t = timesteps[i]

                if self.interrupt:
                    continue

                self._current_timestep = t

                def get_preds(latents, embeds, ids, model_guidance):
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    noise_pred = self.transformer(
                        hidden_states=latents.to(self.transformer.dtype),
                        timestep=timestep.to(self.transformer.dtype) / 1000,
                        guidance=model_guidance,
                        encoder_hidden_states=embeds,
                        txt_ids=ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1), :].to(device=latents.device, dtype=torch.float32)
                    v_t = -noise_pred
                    g2score = 2 * v_t - 2 * latents / (1 - sigmas[i])
                    drift = g2score + latents / (1 - sigmas[i] + eps)
                    return noise_pred, drift, g2score

                if resample:
                    brownian_int, iid_noise = self.sample_brownian_integral(
                        sigmas, num_inference_steps, image_latents.shape
                    )
                source_latents = brownian_int[i] * (1 - sigmas[i]) + image_latents * (1 - sigmas[i])

                _, source_drift, g2score_source = get_preds(
                    source_latents, source_prompt_embeds, source_text_ids, source_guidance
                )
                latents = diff_latents + source_latents
                _, drift, _ = get_preds(latents, prompt_embeds, text_ids, guidance)

                latents_dtype = latents.dtype
                delta_t = sigmas[i] - sigmas[i + 1]
                diffusion_coeff = (2 * sigmas[i] / (1 - sigmas[i] + eps) * delta_t).sqrt()
                _ = g2score_source * delta_t + iid_noise[i] * diffusion_coeff

                diff_latents = diff_latents + (drift - source_drift) * delta_t
                if resample:
                    source_latents = image_latents * (1 - sigmas[i + 1]) + sigmas[i + 1] * torch.randn_like(
                        image_latents
                    )
                else:
                    source_latents = brownian_int[i + 1] * (1 - sigmas[i + 1] + eps) + image_latents * (
                        1 - sigmas[i + 1] + eps
                    )
                latents = diff_latents + source_latents

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean
            latents = self._unpatchify_latents(latents).to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
