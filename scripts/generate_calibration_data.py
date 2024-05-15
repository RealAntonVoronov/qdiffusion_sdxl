import gc
from contextlib import nullcontext

import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from tqdm import trange

cali_data_path = '../sd_coco-s75_sample1024_allst.pt'
sd_calibration_data = torch.load(cali_data_path)
prompts = sd_calibration_data['prompts']
del sd_calibration_data
gc.collect()

xs, ts, res_prompt_embeds, res_add_text_embeds = [], [], [], []

batch_size = 4
seed = 59
guidance_scale = 7.5
num_inference_steps = 50
eta = 0.

model_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_path, 
                                                 torch_dtype=torch.float16, variant="fp16", 
                                                 use_safetensors=True,
                                                 scheduler=DDIMScheduler.from_config(model_path, subfolder="scheduler"),
                                                 ).to("cuda")

# warmup sample to turn on some of the sdxl arguments
pipe("tree")

for prompt_id in trange(0, len(prompts), batch_size):
    cur_prompts = prompts[prompt_id:prompt_id+batch_size]
    cur_xs = []
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    device = generator.device

    # 0. Default height and width to unet
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor
    
    original_size = (height, width)
    target_size = (height, width)
    
    # 1. Check inputs. SKIP
    
    # 2. Define call parameters
    num_images_per_prompt = 1

    # 3. Encode input prompt
    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(prompt=cur_prompts, 
                                                                                                                        device=device)

    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
        
    with torch.inference_mode():
        # 5. Prepare latent variables
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pipe.text_encoder.dtype,
            device,
            generator,
        )
        
        # 6. Prepare extra step kwargs
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    
        add_time_ids = pipe._get_add_time_ids(
                original_size,
                (0, 0),
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        negative_add_time_ids = add_time_ids
    
        # do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        res_prompt_embeds.append(prompt_embeds.cpu())
        res_add_text_embeds.append(add_text_embeds.cpu())
    
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        
        # 8. num_warmup_steps (never used -> SKIP)
        
        # 8.1 Apply denoising_end
        # actually doesn't do anything, so SKIP
        
        # 9. Guidance Scale Embedding is not applied
        timestep_cond = None

        for i, t in enumerate(timesteps):
            # expand the latents since we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    
            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            cur_xs.append(latent_model_input.cpu())
            
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
    
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
            # compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    xs.append(torch.stack(cur_xs))

torch.save({"xs": torch.cat(xs, dim=1), "ts": timesteps.cpu(), "prompt_embeds": torch.cat(res_prompt_embeds),
            "add_text_embeds": torch.cat(res_add_text_embeds), "add_time_ids": add_time_ids[0].cpu(), 
            "prompts": prompts},
            "../sdxl_coco_calibration_data.pt")
