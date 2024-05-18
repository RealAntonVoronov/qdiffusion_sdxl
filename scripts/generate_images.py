import os
from argparse import ArgumentParser
from itertools import chain
from contextlib import nullcontext

import torch
torch.set_num_threads(32)
import numpy as np
import pandas as pd
import wandb
from PIL import Image
from tqdm import tqdm, trange
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils import make_image_grid

from ldm.modules.diffusionmodules.sdxl_unet import QDiffusionUNet
from qdiff.quant_model import QuantModel
from qdiff.utils import resume_cali_model


def prepare_prompts(prompts_path):
    bs = 1
    world_size = 2
    
    df = pd.read_csv(prompts_path)
    all_text = list(df["captions"])

    num_batches = ((len(all_text) - 1) // (bs * world_size) + 1) * world_size
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = []
    for i in range(world_size):
        rank_batches.append(all_batches[i :: world_size])

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = [all_batches_index[i :: world_size] for i in range(world_size)]
    
    return rank_batches, rank_batches_index, all_text


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cali_ckpt", required=True, ) # 'quantized_models/debug/2024-05-16-17-14-29/ckpt.pth'
    parser.add_argument("--out_path", required=True, ) # 'quantized_models/debug/2024-05-16-17-14-29/sbs_images'
    parser.add_argument("--eval_prompts_path", default="eval_prompts/parti-prompts-eval.csv")
    parser.add_argument("--weight_bit", type=int, default=4)
    parser.add_argument("--act_bit", type=int, default=32)
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--sdxl_path", default="stabilityai/stable-diffusion-xl-base-1.0")
    args = parser.parse_args()
    return args


def load_quantized_unet(ckpt_path, weight_bit=4, act_bit=32, device='cuda'):
    sdxl_path = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = QDiffusionUNet.from_pretrained(sdxl_path, use_safetensors=True,
                                          subfolder='unet',
                                          ).to(device)
    setattr(unet, 'split', False)

    wq_params = {'n_bits': weight_bit, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': act_bit, 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  False}
    qnn = QuantModel(model=unet, weight_quant_params=wq_params, act_quant_params=aq_params,
                     act_quant_mode="qdiff", sm_abit=act_bit)
    qnn.to(device)
    qnn.eval()
    qnn.set_grad_ckpt(False)
    # required for 1 run to correctly load parameters
    cali_data = (torch.randn(1, 4, 128, 128), torch.randint(0, 1000, (1,)), 
                 torch.randn(1, 77, 2048), torch.randn(1, 1280), torch.tensor([[1024, 1024, 0, 0, 1024, 1024]]))
    cali_data = [x.to(unet.dtype) for x in cali_data]
    resume_cali_model(qnn, ckpt_path, cali_data, False, "qdiff", cond=True, sdxl=True)
    torch.cuda.empty_cache()
    
    return unet


def generate_with_quantized_sdxl(pipe, prompt, num_images_per_prompt=1, output_type='pt',
                                 device='cuda', seed=59, guidance_scale=5, num_inference_steps=50, disable_tqdm=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    device = generator.device

    # 0. Default height and width to unet
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor
    
    original_size = (height, width)
    target_size = (height, width)
    
    # 1. Check inputs. SKIP

    # 2 define call parameters
    if isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = 1

    # 3. get prompt embeddings
    (prompt_embeds, negative_prompt_embeds, 
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(prompt=prompt, 
                                                                               num_images_per_prompt=num_images_per_prompt,
                                                                               device=device,
                                                                               )
    
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
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, 0.)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim
    
        add_time_ids = list(original_size + (0, 0) + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype)
        negative_add_time_ids = add_time_ids

        # do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. num_warmup_steps (never used -> SKIP)
        
        # 8.1 Apply denoising_end
        # actually doesn't do anything, so SKIP
        
        # 9. Guidance Scale Embedding is not applied
        timestep_cond = None
            
        progress_bar = trange(num_inference_steps) if not disable_tqdm else nullcontext()
        with progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents since we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
        
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    context=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                )
                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if not disable_tqdm:
                    progress_bar.update()

        needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        if needs_upcasting:
            pipe.upcast_vae()
            latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
        # cast back to fp16 if needed
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        
        return pipe.image_processor.postprocess(image, output_type=output_type)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.out_path, exist_ok=True)

    if args.debug:
        prompts = pd.read_csv("eval_prompts/parti-prompts-eval.csv")['captions'].tolist()[:8]
    else:
        rank_eval_prompts, rank_eval_prompts_indices, all_text = prepare_prompts(args.eval_prompts_path)

    if args.debug:
        # load model
        sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(args.sdxl_path, use_safetensors=True,
                                                                  scheduler=DDIMScheduler.from_config(args.sdxl_path, subfolder="scheduler"),
                                                                  ).to(device)
        # load quantized unet
        unet = load_quantized_unet(args.cali_ckpt, weight_bit=args.weight_bit, act_bit=args.act_bit, device=device)
        torch.cuda.empty_cache()
    else:
        # load quantized unet
        unet = load_quantized_unet(args.cali_ckpt, weight_bit=args.weight_bit, act_bit=args.act_bit, device=device)
        torch.cuda.empty_cache()

        # load model
        sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(args.sdxl_path, use_safetensors=True,
                                                                  scheduler=DDIMScheduler.from_config(args.sdxl_path, subfolder="scheduler"),
                                                                  ).to(device)
    
    # change pipelines' unet to a quantized one
    sdxl_pipeline.unet = unet

    if args.debug:
        res_images = {'teacher': []}
        for image in sorted(os.listdir('teacher_imgs')):
            if image.endswith('.jpg'):
                res_images['teacher'].append(Image.open(f'teacher_imgs/{image}'))
        res_images['student'] = generate_with_quantized_sdxl(sdxl_pipeline, prompt=prompts, output_type='pil', device=device, disable_tqdm=False,
                                                             seed=42, num_images_per_prompt=1)
        
        images_1 = res_images['teacher']
        images_2 = res_images['student']

        res_grid = make_image_grid(list(chain.from_iterable(zip(images_1, images_2))), rows=len(images_1), cols=2, resize=512)
        images = wandb.Image(res_grid, caption="Left: Teacher, Right: Student")
        wandb.log({"examples": images}, step=0)
    else:
        for i, prompts in enumerate(rank_eval_prompts):
            for seed, batch in enumerate(tqdm(prompts)):
                images = generate_with_quantized_sdxl(sdxl_pipeline, prompt=list(batch), output_type='pil', device=device, disable_tqdm=True,
                                                    seed=seed, num_images_per_prompt=args.num_images_per_prompt)
                for j, image in enumerate(images):
                    image.save(f"{args.out_path}/{rank_eval_prompts_indices[i][seed][0]}_{j}.jpg")



if __name__ == '__main__':
    main()