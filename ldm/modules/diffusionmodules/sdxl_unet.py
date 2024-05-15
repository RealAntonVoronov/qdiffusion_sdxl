from diffusers import UNet2DConditionModel
import time
import logging
logger = logging.getLogger(__name__)

class QDiffusionUNet(UNet2DConditionModel):
    def forward(self, x, timesteps=None, context=None, added_cond_kwargs=None, debug=False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # 1. time
        t0 = time.time()
        if debug:
            print("timestep embedding")
        timestep_cond = None
        t_emb = self.get_time_embed(sample=x, timestep=timesteps)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        # if debug:
        #     print('context:', context)
        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs,
        )

        emb = emb + aug_emb if aug_emb is not None else emb
        if debug:
            print(f"timestep embedding took {time.time() - t0}s")
        # 2. pre-process
        if debug:
            print("step 2. pre-process")
        t0 = time.time()
        sample = self.conv_in(x)
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=context, added_cond_kwargs=added_cond_kwargs,
        )
        # if debug:
        #     print('encoder_hidden_states after self.process_encoder_hidden_states', encoder_hidden_states)
        if debug:
            print(f"step 2. pre-process took {time.time() - t0}s")

        # 3. down
        down_block_res_samples = (sample,)
        # import ipdb; ipdb.set_trace()
        for i, downsample_block in enumerate(self.down_blocks):
            t0 = time.time()
            if debug:
                print(f"step 3. down; downsample_block: {i}")
            
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
                print(f"step 3.{i} took {time.time() - t0}s")
        # 4. mid
        if self.mid_block is not None:
            t0 = time.time()
            if debug:
                print("step 4. mid block")
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
                print(f"step 4 took {time.time() - t0}s")
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if debug:
                print(f"step 5. up; upsample_block: {i}")
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
                print(f"step 5.{i} took {time.time() - t0}s")

        # 6. post-process
        if debug:
            print("step 6. post-process")
        t0 = time.time()
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if debug:
            print(f"step 6 took {time.time() - t0}s")

        return sample
