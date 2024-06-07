from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_blocks import UpBlock2D, CrossAttnUpBlock2D
import time
import logging

import torch
from torch import nn
logger = logging.getLogger(__name__)

class CustomUpBlock2D(nn.Module):
    def __init__(self, orig_block, split=False):
        super().__init__()
        self.resnets = orig_block.resnets
        self.upsamplers = orig_block.upsamplers
        self.resolution_idx = orig_block.resolution_idx
        self.split = split

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if self.split:
                split = hidden_states.shape[1]
            else:
                split = 0
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb, split=split)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class CustomCrossAttnUpBlock2D(nn.Module):
    def __init__(self, orig_block, split=False):
        super().__init__()
        self.has_cross_attention = orig_block.has_cross_attention
        self.num_attention_heads = orig_block.num_attention_heads

        self.attentions = orig_block.attentions
        self.resnets = orig_block.resnets
        self.upsamplers = orig_block.upsamplers

        self.gradient_checkpointing = False
        self.resolution_idx = orig_block.resolution_idx
        self.split = split

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, encoder_hidden_states=None,
                cross_attention_kwargs=None, upsample_size=None,
                attention_mask=None, encoder_attention_mask=None,
                ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if self.split:
                split = hidden_states.shape[1]
            else:
                split = 0

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb, split=split)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class QDiffusionUNet(UNet2DConditionModel):
    def block_refactor(self):
        self.up_blocks[0] = CustomCrossAttnUpBlock2D(self.up_blocks[0], split=self.split)
        self.up_blocks[1] = CustomCrossAttnUpBlock2D(self.up_blocks[1], split=self.split)
        self.up_blocks[2] = CustomUpBlock2D(self.up_blocks[2], split=self.split)

    def forward(self, x, timesteps=None, context=None, added_cond_kwargs=None, debug=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # cast inputs to unet dtype
        x = x.to(self.dtype)
        timesteps = timesteps.to(self.dtype)
        context = context.to(self.dtype)
        added_cond_kwargs = {k: v.to(self.dtype) for k, v in added_cond_kwargs.items()}

        # 1. time
        t0 = time.time()
        if debug:
            logger.info("timestep embedding")
        timestep_cond = None
        t_emb = self.get_time_embed(sample=x, timestep=timesteps)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        # if debug:
        #     logger.info('context:', context)
        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs,
        )

        emb = emb + aug_emb if aug_emb is not None else emb
        if debug:
            logger.info(f"timestep embedding took {time.time() - t0}s")
        # 2. pre-process
        if debug:
            logger.info("step 2. pre-process")
        t0 = time.time()
        sample = self.conv_in(x)
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs,
        )
        # if debug:
        #     logger.info('encoder_hidden_states after self.process_encoder_hidden_states', encoder_hidden_states)
        if debug:
            logger.info(f"step 2. pre-process took {time.time() - t0}s")

        # 3. down
        down_block_res_samples = (sample,)
        # import ipdb; ipdb.set_trace()
        for i, downsample_block in enumerate(self.down_blocks):
            t0 = time.time()
            if debug:
                logger.info(f"step 3. down; downsample_block: {i}")
            
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    encoder_attention_mask=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
            down_block_res_samples += res_samples
            if debug:
                logger.info(f"step 3.{i} took {time.time() - t0}s")
        # 4. mid
        if self.mid_block is not None:
            t0 = time.time()
            if debug:
                logger.info("step 4. mid block")
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                    encoder_attention_mask=None,
                )
            else:
                sample = self.mid_block(sample, emb)
            if debug:
                logger.info(f"step 4 took {time.time() - t0}s")
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if debug:
                logger.info(f"step 5. up; upsample_block: {i}")
            t0 = time.time()
            if self.split:
                split = sample.shape[1]
            else:
                split = 0

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=None,
                    upsample_size=None,
                    attention_mask=None,
                    encoder_attention_mask=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=None,
                )
            if debug:
                logger.info(f"step 5.{i} took {time.time() - t0}s")

        # 6. post-process
        if debug:
            logger.info("step 6. post-process")
        t0 = time.time()
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if debug:
            logger.info(f"step 6 took {time.time() - t0}s")

        return sample
