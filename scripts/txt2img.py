import argparse, os, datetime, gc, yaml, time
import logging
from itertools import islice, chain
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(32)
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils import make_image_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.diffusionmodules.sdxl_unet import QDiffusionUNet
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from scripts.generate_images import generate_with_quantized_sdxl
try:
    import nirvana_dl
except ImportError:
    print('NO NIRVANA DL PACKAGE')
    nirvana_dl = None

logger = logging.getLogger(__name__)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_data_path", type=str, 
        # default="sd_coco_sample1024_allst.pt",
        help="calibration dataset path",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--sdxl", action="store_true",
        help="run q-diffusion on SDXL UNet",
    )
    parser.add_argument(
        "--scale_method", choices=["max", "mse"], default="mse",
        help="quantization initialization method. 'max' is fast, 'mse' is used in original work.",
    )
    parser.add_argument(
        "--cali_data_size", type=int, default=6400,
        help="use less data for calibration than SD v1.5 to be able to run it in a finite time",
    )
    parser.add_argument(
        "--save_after_init", action='store_true',
        help="save checkpoint after quantization paramters initailization. Might be valuable for mse init."
    )
    parser.add_argument(
        "--exp_name", default='sdxl_w4a32',
    )
    parser.add_argument(
        "--sdxl_path", default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp32')
    opt = parser.parse_args()

    if opt.precision == 'fp16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if opt.exp_name is None:
        opt.exp_name = f"{opt.scale_method}_init_s{opt.cali_data_size}_iters{opt.cali_iters}"
    wandb.init(entity='rock-and-roll', project='baselines', name=opt.exp_name)

    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if not opt.sdxl:
        config = OmegaConf.load(f"{opt.config}")
        model = load_model_from_config(config, f"{opt.ckpt}")
        model = model.to(device)
        if opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
        unet = sampler.model.model.diffusion_model
    else:
        # torch_dtype = torch.float16 if opt.sdxl_fp16 else torch.float32
        # variant = "fp16" if opt.sdxl_fp16 else None
        sdxl_path = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = QDiffusionUNet.from_pretrained(sdxl_path, use_safetensors=True,
                                              # torch_dtype=torch_dtype, variant=variant,
                                              subfolder='unet',
                                              ).to(device)

    # ptq
    if opt.split:
        setattr(unet, "split", True)

    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': opt.scale_method}
    aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': opt.scale_method, 'leaf_param':  opt.quant_act}
    if opt.resume:
        logger.info('Load with min-max quick initialization')
        wq_params['scale_method'] = 'max'
        aq_params['scale_method'] = 'max'
    if opt.resume_w:
        wq_params['scale_method'] = 'max'
    qnn = QuantModel(
        model=unet, weight_quant_params=wq_params, act_quant_params=aq_params,
        act_quant_mode="qdiff", sm_abit=opt.sm_abit)
    qnn.cuda()
    qnn.eval()

    if opt.no_grad_ckpt:
        logger.info('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)

    if opt.resume:
        if opt.sdxl:
            cali_data = (torch.randn(1, 4, 128, 128), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 2048), torch.randn(1, 1280), torch.tensor([[1024, 1024, 0, 0, 1024, 1024]]))
            cali_data = [x.to(unet.dtype) for x in cali_data]
        else:
            cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
        resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=True, sdxl=opt.sdxl)
    else:
        logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
        if opt.cali_data_path:
            sample_data = torch.load(opt.cali_data_path)
            cali_data = get_train_samples(opt, sample_data, opt.ddim_steps, sdxl=opt.sdxl)
            cali_data = [x[:opt.cali_data_size].to(unet.dtype) for x in cali_data]
            del(sample_data)
            gc.collect()
            logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")
        else:
            cali_data = (torch.randn(6400, 4, 128, 128), torch.randint(0, 1000, (6400,)), torch.randn(6400, 77, 2048), torch.randn(6400, 1280), 
                            torch.tensor([[1024, 1024, 0, 0, 1024, 1024]]).repeat(6400, 1))
            cali_data = [x.to(unet.dtype) for x in cali_data]

        if opt.sdxl:
            cali_xs, cali_ts, cali_cs, cali_cs_pooled, cali_add_time_ids = cali_data
        else:    
            cali_xs, cali_ts, cali_cs = cali_data
        if opt.resume_w:
            resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond)
        else:
            logger.info("Initializing weight quantization parameters")
            qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
            init_batch_size = 5
            if opt.sdxl:
                added_cond_kwargs = {"text_embeds": cali_cs_pooled[:init_batch_size].cuda(), "time_ids": cali_add_time_ids[:init_batch_size].cuda()}
            else:
                added_cond_kwargs = {}

            with torch.no_grad():
                _ = qnn(cali_xs[:init_batch_size].cuda(), cali_ts[:init_batch_size].cuda(), cali_cs[:init_batch_size].cuda(), 
                        added_cond_kwargs=added_cond_kwargs, debug=True,
                        )
            logger.info("Initializing has done!")
            if opt.save_after_init:
                logger.info("Saving calibrated quantized UNet model")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt_inited.pth"))
                if nirvana_dl is not None:
                    nirvana_dl.snapshot.dump_snapshot()

        # Kwargs for weight rounding calibration
        kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                    iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                    warmup=0.2, act_quant=False, opt_mode='mse', cond=True, sdxl=opt.sdxl)
        
        def recon_model(model):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                if name == 'output_blocks':
                    logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                    in_recon_done = True
                    torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                    if nirvana_dl is not None:
                        nirvana_dl.snapshot.dump_snapshot()
                if name.isdigit() and int(name) >= 9:
                    logger.info(f"Saving temporary checkpoint at {name}...")
                    torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                    if nirvana_dl is not None:
                        nirvana_dl.snapshot.dump_snapshot()
                    
                if isinstance(module, QuantModule):
                    if module.ignore_reconstruction is True:
                        logger.info('Ignore reconstruction of layer {}'.format(name))
                        continue
                    else:
                        logger.info('Reconstruction for layer {}'.format(name))
                        layer_reconstruction(qnn, module, **kwargs)
                elif isinstance(module, BaseQuantBlock):
                    if module.ignore_reconstruction is True:
                        logger.info('Ignore reconstruction of block {}'.format(name))
                        continue
                    else:
                        logger.info('Reconstruction for block {}'.format(name))
                        block_reconstruction(qnn, module, with_kwargs=True, **kwargs)
                else:
                    recon_model(module)

        if not opt.resume_w:
            logger.info("Doing weight calibration")
            recon_model(qnn)
            qnn.set_quant_state(weight_quant=True, act_quant=False)
        if opt.quant_act:
            logger.info("UNet model")
            logger.info(unet)
            logger.info("Doing activation calibration")
            # Initialize activation quantization parameters
            qnn.set_quant_state(True, True)
            with torch.no_grad():
                batch_size = opt.cali_batch_size
                inds = np.random.choice(cali_xs.shape[0], batch_size, replace=False)
                if opt.sdxl:
                    added_cond_kwargs = {"text_embeds": cali_cs_pooled[inds].cuda(), "time_ids": cali_add_time_ids[inds].cuda()}
                else:
                    added_cond_kwargs = {}
                _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda(), cali_cs[inds].cuda(), added_cond_kwargs=added_cond_kwargs)
                if opt.running_stat:
                    logger.info('Running stat for activation quantization')
                    inds = np.arange(cali_xs.shape[0])
                    np.random.shuffle(inds)
                    qnn.set_running_stat(True, opt.rs_sm_only)
                    for i in trange(int(cali_xs.size(0) / batch_size)):
                        if opt.sdxl:
                            added_cond_kwargs = {"text_embeds": cali_cs_pooled[inds[i * batch_size:(i + 1) * batch_size]].cuda(), 
                                                    "time_ids": cali_add_time_ids[inds[i * batch_size:(i + 1) * batch_size]].cuda()}
                        else:
                            added_cond_kwargs = {}
                        _ = qnn(cali_xs[inds[i * batch_size:(i + 1) * batch_size]].cuda(), 
                            cali_ts[inds[i * batch_size:(i + 1) * batch_size]].cuda(),
                            cali_cs[inds[i * batch_size:(i + 1) * batch_size]].cuda(),
                            added_cond_kwargs)
                    qnn.set_running_stat(False, opt.rs_sm_only)

            kwargs = dict(
                cali_data=cali_data, 
                batch_size=opt.cali_batch_size, iters=opt.cali_iters_a, act_quant=True, 
                opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=True)
            recon_model(qnn)
            qnn.set_quant_state(weight_quant=True, act_quant=True)
        
        logger.info("Saving calibrated quantized UNet model")
        for m in qnn.model.modules():
            if isinstance(m, AdaRoundQuantizer):
                m.zero_point = nn.Parameter(m.zero_point)
                m.delta = nn.Parameter(m.delta)
            elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                if m.zero_point is not None:
                    if not torch.is_tensor(m.zero_point):
                        m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                    else:
                        m.zero_point = nn.Parameter(m.zero_point)
        torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
        if nirvana_dl is not None:
            nirvana_dl.snapshot.dump_snapshot()

    logger.info('sampling')
    if not opt.sdxl:
        sampler.model.model.diffusion_model = qnn
    else:
        torch.cuda.empty_cache()
        sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(opt.sdxl_path,
                                                                  torch_dtype=torch_dtype, variant='fp16', 
                                                                  use_safetensors=True,
                                                                  scheduler=DDIMScheduler.from_config(opt.sdxl_path, subfolder="scheduler"),
                                                                  ).to(device)
        sdxl_pipeline.unet = qnn

    captions = pd.read_csv('eval_prompts/parti-prompts-eval.csv')["captions"].tolist()[:8]
    res_images = {'teacher': []}
    for image in sorted(os.listdir('teacher_imgs')):
        if image.endswith('.jpg'):
            res_images['teacher'].append(Image.open(f'teacher_imgs/{image}'))
    res_images['student'] = generate_with_quantized_sdxl(sdxl_pipeline, prompt=captions, output_type='pil', device=device, disable_tqdm=False,
                                                         seed=opt.seed, num_images_per_prompt=1)
    
    images_1 = res_images['teacher']
    images_2 = res_images['student']

    res_grid = make_image_grid(list(chain.from_iterable(zip(images_1, images_2))), rows=len(images_1), cols=2, resize=512)
    images = wandb.Image(res_grid, caption="Left: Teacher, Right: Student")
    wandb.log({"examples": images}, step=0)
    res_grid.save(f"{outpath}/grid.jpg")
    for i, image in enumerate(res_images['student']):
        image.save(f'{outpath}/{i}.jpg')
    if nirvana_dl is not None:
        nirvana_dl.snapshot.dump_snapshot()
    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
    wandb.finish()

    
if __name__ == "__main__":
    main()
