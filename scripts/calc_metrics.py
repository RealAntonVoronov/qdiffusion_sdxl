import argparse
import datetime
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from itertools import chain

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(32)
import torch.distributed as dist
import wandb
from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg, retrieve_timesteps
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from scripts.generate_images import get_checkpoint_path, load_quantized_unet, generate_with_quantized_sdxl
from src.fid_score_in_memory import calculate_fid




def dist_init():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"

    backend = "gloo" if not torch.cuda.is_available() else "nccl"
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(0, 3600))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def prepare_prompts(args):
    df = pd.read_csv(args.evaluation_prompts)
    all_text = list(df["captions"])
    all_text = all_text[: args.max_count]

    num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank() :: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


def _inference_loop(
    pipeline: StableDiffusionXLPipeline,
    latents: torch.Tensor,
    timesteps: List[int],
    num_inference_steps: int,
    unet_kwargs: dict,
    extra_step_kwargs: dict,
    return_intermediate_latents: bool,
) -> torch.Tensor:
    """Core Unet inference loop that runs each time you call do_inference below"""
    pipeline._num_timesteps = len(timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * pipeline.scheduler.order, 0)
    intermediate_latents = []

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        progress_bar.desc = "Inference loop"
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue
            if return_intermediate_latents:
                intermediate_latents.append((t, latents, unet_kwargs))

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if pipeline.do_classifier_free_guidance else latents

            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual

            with torch.inference_mode():
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    **unet_kwargs,
                    return_dict=False,
                )

            # perform guidance
            if pipeline.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if pipeline.do_classifier_free_guidance and pipeline.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipeline.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
    return latents if not return_intermediate_latents else intermediate_latents

def do_inference(
    pipeline: StableDiffusionXLPipeline,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    device: Optional[torch.device] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    return_intermediate_latents: bool = False,
    **kwargs,
):
    r"""
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
            Anything below 512 pixels won't work well for
            [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
            and checkpoints that are not specifically fine-tuned on low resolutions.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        denoising_end (`float`, *optional*):
            When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
            completed before it is intentionally prematurely terminated. As a result, the returned sample will
            still retain a substantial amount of noise as determined by the discrete timesteps selected by the
            scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
            "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
            Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
        guidance_scale (`float`, *optional*, defaults to 5.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
            of a plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
            [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
            Guidance rescale factor should fix overexposure when using zero terminal SNR.
        original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
        negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a specific image resolution. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
            micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a target image resolution. It should be as same
            as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
            [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
            information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """
    callback_steps = kwargs.pop("callback_steps", None)
    if return_intermediate_latents:
        assert output_type == "latent"

    # 0. Default height and width to unet
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    pipeline._guidance_scale = guidance_scale
    pipeline._guidance_rescale = guidance_rescale
    pipeline._clip_skip = clip_skip
    pipeline._cross_attention_kwargs = cross_attention_kwargs
    pipeline._denoising_end = denoising_end
    pipeline._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if device is None:
        device = getattr(pipeline, "_execution_device", None)
    if device is None:
        device = next(chain(pipeline.unet.parameters(), pipeline.unet.buffers(), (None,))).device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Encode input prompt
    lora_scale = (
        pipeline.cross_attention_kwargs.get("scale", None) if pipeline.cross_attention_kwargs is not None else None
    )

    with torch.inference_mode():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=pipeline.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=pipeline.clip_skip,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipeline._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if pipeline.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None:
        assert False, "removed from prototype"

    # 8. Denoising loop
    # 8.1 Apply denoising_end
    if (
        pipeline.denoising_end is not None
        and isinstance(pipeline.denoising_end, float)
        and pipeline.denoising_end > 0
        and pipeline.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                pipeline.scheduler.config.num_train_timesteps
                - (pipeline.denoising_end * pipeline.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if pipeline.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(pipeline.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = pipeline.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipeline.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    unet_kwargs = dict(
        context=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=pipeline.cross_attention_kwargs,
        added_cond_kwargs=added_cond_kwargs,
    )

    latents = _inference_loop(
        pipeline,
        latents,
        timesteps,
        num_inference_steps,
        unet_kwargs=unet_kwargs,
        extra_step_kwargs=extra_step_kwargs,
        return_intermediate_latents=return_intermediate_latents,
    )

    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

        if needs_upcasting:
            latents = latents.to(pipeline.vae.dtype)
            pipeline.upcast_vae()

        image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            pipeline.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        # apply watermark if available
        if pipeline.watermark is not None:
            image = pipeline.watermark.apply_watermark(image)

        image = pipeline.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    pipeline.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return StableDiffusionXLPipelineOutput(images=image)


@torch.no_grad()
def distributed_sampling(pipeline, device, args):

    pipeline.set_progress_bar_config(disable=True)

    pipeline = pipeline.to(device)

    rank_batches, rank_batches_index, all_prompts = prepare_prompts(args)

    local_images = []
    local_text_idxs = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for cnt, mini_batch in enumerate(tqdm(rank_batches, unit="batch", disable=(dist.get_rank() != 0))):
        if args.generate_teacher:
            images = pipeline(prompt=list(mini_batch), output_type='pil', device=device, 
                              generator=generator, num_images_per_prompt=args.num_images_per_prompt,
                              guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                              ).images
        elif not args.vahe_code:
            images = generate_with_quantized_sdxl(pipeline, prompt=list(mini_batch), output_type='pil', device=device, 
                                                  disable_tqdm=True, generator=generator, num_images_per_prompt=args.num_images_per_prompt,
                                                  guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                                                  )
        else:
            images = do_inference(pipeline,
                    prompt=list(mini_batch),
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=args.num_images_per_prompt,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx]))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()

    gathered_images = [torch.zeros_like(local_images) for _ in range(dist.get_world_size())]
    gathered_text_idxs = [torch.zeros_like(local_text_idxs) for _ in range(dist.get_world_size())]

    dist.all_gather(gathered_images, local_images)  # gather not supported with NCCL
    dist.all_gather(gathered_text_idxs, local_text_idxs)

    images, prompts = [], []
    if dist.get_rank() == 0:
        gathered_images = np.concatenate([images.cpu().numpy() for images in gathered_images], axis=0)
        gathered_text_idxs = np.concatenate([text_idxs.cpu().numpy() for text_idxs in gathered_text_idxs], axis=0)
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            images.append(ToPILImage()(image))
            prompts.append(all_prompts[global_idx])
    # Done.
    dist.barrier()
    return images, prompts


@torch.no_grad()
def calc_pick_and_clip_scores(model, image_inputs, text_inputs, batch_size=50):
    assert len(image_inputs) == len(text_inputs)

    scores = torch.zeros(len(text_inputs))
    for i in range(0, len(text_inputs), batch_size):
        image_batch = image_inputs[i : i + batch_size]
        text_batch = text_inputs[i : i + batch_size]
        # embed
        with torch.cuda.amp.autocast():
            image_embs = model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.cuda.amp.autocast():
            text_embs = model.get_text_features(text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i : i + batch_size] = (text_embs * image_embs).sum(-1)  # model.logit_scale.exp() *
    return scores.cpu()


def calculate_scores(args, images, prompts, device="cuda"):
    processor = AutoProcessor.from_pretrained(args.clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(args.clip_model_name_or_path).eval().to(device)
    pickscore_model = AutoModel.from_pretrained(args.pickscore_model_name_or_path).eval().to(device)

    image_inputs = processor(images=images, return_tensors="pt",)[
        "pixel_values"
    ].to(device)

    text_inputs = processor(text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt",)[
        "input_ids"
    ].to(device)

    print("Evaluating PickScore...")
    pick_score = calc_pick_and_clip_scores(pickscore_model, image_inputs, text_inputs).mean()
    print("Evaluating CLIP ViT-H-14 score...")
    clip_score = calc_pick_and_clip_scores(clip_model, image_inputs, text_inputs).mean()
    print("Evaluating FID score...")
    fid_score = calculate_fid(
        images, args.coco_ref_stats_path, inception_path=args.inception_path
    )  # https://github.com/yandex-research/lcm/tree/9886452e69931b2520a8ec43540b50acef243ca4/stats
    return pick_score, clip_score, fid_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--out_path", required=True)
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp32')
    parser.add_argument("--exp_name", help="Experiment name to log in WandB")
    # quantization params
    parser.add_argument("--weight_bit", type=int, default=4)
    parser.add_argument("--act_bit", type=int, default=32)
    # models paths
    parser.add_argument("--sdxl_path", default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="path to Stable Diffusion XL Pipeline to load from_pretrained()",
                        )
    parser.add_argument("--quantized_model_path", required=True,
                        help="path to quantized diffusion model to load, as in AutoPipelineForText2Image.from_pretrained()",
                        )
    parser.add_argument("--evaluation_prompts", default="eval_prompts/coco.csv",
                        help="Path to prompts dataset (newline-separated text file)",
                        )
    parser.add_argument("--clip_model_name_or_path",
                        default="${INPUT_PATH}/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        help="path to clip model to load, as in AutoModel.from_pretrained",
                        )
    parser.add_argument("--pickscore_model_name_or_path",
                        default="${INPUT_PATH}/PickScore_v1",
                        help="path to pickscore model to load, as in AutoModel.from_pretrained",
                        )
    parser.add_argument("--coco_ref_stats_path",
                        default="stats/fid_stats_mscoco512_val.npz",
                        help="Path to reference stats from coco",
                        )
    parser.add_argument("--inception_path",
                        default="stats/pt_inception-2015-12-05-6726825d.pth",
                        help="Path to inception reference stats ",
                        )
    # evaluation params
    parser.add_argument("--bs", type=int, default=1, help="Prompt batch size")
    parser.add_argument("--max_count", type=int, default=5000, help="Prompt count to eval on ")
    parser.add_argument('--seed', type=int)
    # diffusion inference params
    parser.add_argument("--num_inference_steps", type=int, default=50, help="number of inference steps used for calibration and evaluation")
    parser.add_argument("--guidance_scale", type=float, default=5, help="Guidance scale as defined in [Classifier-Free Diffusion Guidance]")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--generate_teacher", action="store_true")
    parser.add_argument("--vahe_code", action="store_true")

    args = parser.parse_args()
    args.num_images_per_prompt = 1

    if args.precision == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    dist_init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dist.get_rank() == 0:
        os.makedirs(args.out_path, exist_ok=True)

    # load quantized unet
    if not args.generate_teacher:
        ckpt_path = get_checkpoint_path(args.quantized_model_path)
        unet = load_quantized_unet(ckpt_path, weight_bit=args.weight_bit, act_bit=args.act_bit, device=device, split=args.split)
        setattr(unet.add_embedding.linear_1, 'in_features', 2816)
        torch.cuda.empty_cache()

    # load model
    sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(args.sdxl_path, use_safetensors=True,
                                                              torch_dtype=torch_dtype, variant='fp16',
                                                              scheduler=DDIMScheduler.from_config(args.sdxl_path, subfolder="scheduler"),
                                                              ).to(device)
    
    # change pipelines' unet to a quantized one
    if not args.generate_teacher:
        sdxl_pipeline.unet = unet

    # if dist.get_rank() == 0:
    #     wandb.init(entity='rock-and-roll', project='baselines', name=args.exp_name)

    print("Generating with a quantized model.")
    images, prompts = distributed_sampling(sdxl_pipeline, device, args)
    if dist.get_rank() == 0:
        for i, image in enumerate(images):
            image.save(os.path.join(args.out_path, f"{i:04d}.jpg"))
        with open(os.path.join(args.out_path, 'prompts.txt'), 'w') as f:
            f.writelines('\n'.join(prompts))

    if dist.get_rank() == 0:
        pick_score, clip_score, fid_score = calculate_scores(args, images, prompts, device=device)
        # wandb.log({"pick_score": pick_score, "clip_score": clip_score, "fid_score": fid_score})
        print(f"{pick_score}", f"{clip_score}", f"{fid_score}")