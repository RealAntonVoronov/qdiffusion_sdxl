import argparse
import datetime
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(32)
import torch.distributed as dist
import wandb
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
        else:
            images = generate_with_quantized_sdxl(pipeline, prompt=list(mini_batch), output_type='pil', device=device, 
                                                  disable_tqdm=True, generator=generator, num_images_per_prompt=args.num_images_per_prompt,
                                                  guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps,
                                                  )
        # print("generating with Vahe's code")
        # images = do_inference(pipeline,
        #         prompt=list(mini_batch),
        #         num_inference_steps=args.num_inference_steps,
        #         num_images_per_prompt=args.num_images_per_prompt,
        #         guidance_scale=args.guidance_scale,
        #         generator=generator,
        #     ).images

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
        torch.cuda.empty_cache()

    # load model
    sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(args.sdxl_path, use_safetensors=True,
                                                              torch_dtype=torch_dtype, variant='fp16',
                                                              scheduler=DDIMScheduler.from_config(args.sdxl_path, subfolder="scheduler"),
                                                              ).to(device)
    
    # change pipelines' unet to a quantized one
    if not args.generate_teacher:
        sdxl_pipeline.unet = unet

    if dist.get_rank() == 0:
        wandb.init(entity='rock-and-roll', project='baselines', name=args.exp_name)

    print("Generating with a quantized model.")
    images, prompts = distributed_sampling(sdxl_pipeline, device, args)
    if dist.get_rank() == 0:
        for i, image in enumerate(images):
            image.save(os.path.join(args.out_path, f"{i:04d}.jpg"))
        with open(os.path.join(args.out_path, 'prompts.txt'), 'w') as f:
            f.writelines('\n'.join(prompts))

    if dist.get_rank() == 0:
        pick_score, clip_score, fid_score = calculate_scores(args, images, prompts, device=device)
        wandb.log({"pick_score": pick_score, "clip_score": clip_score, "fid_score": fid_score})
        print(f"{pick_score}", f"{clip_score}", f"{fid_score}")