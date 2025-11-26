from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from transformers import StoppingCriteria
import torch.nn.functional as F
import numpy as np
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from transformers import AutoConfig, AutoTokenizer
from .image_utils import generate_crop_size_list, var_center_crop
from ..qwen2_vl.prompt import Qwen2VLPromptMixin
import numpy as np
from ..base import BaseModel
from ...smp import get_gpu_memory, listinstr
from ...dataset import DATASET_MODALITY
from .modeling_xllmx_llada import LLaDAForMultiModalGeneration


class Lumina_DiMOO(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self, 
        model_path='Alpha-VLLM/Lumina-DiMOO',
        vae_ckpt='Alpha-VLLM/Lumina-DiMOO',
        use_custom_prompt: bool = False,
        verbose: bool = False,
        gen_length=128,
        block_length=128,
        steps=128,       
        **kwargs
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae")
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cpu"
        ).to("cuda")
        self.model._tp_plan = []
        self.system_prompt = "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer."
        self.gen_length = gen_length
        self.steps = steps
        self.block_length = block_length 
        self.mask_id = 126336
        self.verbose = verbose

    def add_break_line(self, sequence, H, W, new_number=0):
        result = []
        for i in range(H):
            start = i * W
            end = start + W
            row = sequence[start:end]
            result.extend(row + [new_number])
        return result

    def encode_image(self, img):
        orig = img.convert("RGB")
        orig_resized = orig
        vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
        x = image_processor.preprocess(orig_resized)
        latents = self.vqvae.encode(x).latents
        latents_bsz, channels, lat_h, lat_w = latents.shape
        quantized = self.vqvae.quantize(latents)[2][2].reshape(latents_bsz, lat_h, lat_w) + 126356
        quantized_new_line = self.add_break_line(quantized.flatten().tolist(), lat_h, lat_w, new_number=126084)
        image_token = [126349] + quantized_new_line + [126350]
        return image_token

    def _prepare_content(self, message, dataset):
        sys_part = "<system>" + self.system_prompt + "</system>"
        count = sum(item['type'] == 'image' for item in message)
        image_tokens = []
        text = ''
        for m in message:
            if m['type'] == "image":
                image = Image.open(m['value'])
                area = image.size[0] * image.size[1]
                if count > 1:
                    crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
                    image = var_center_crop(image, crop_size_list=crop_size_list)
                    image_token = self.encode_image(image)
                else:
                    if area < 512 *512:
                        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
                    elif area > 1024* 1024:
                        crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
                    else:
                        crop_size_list = [(image.size[0]//16*16,image.size[1]//16*16)]
                    image = var_center_crop(image, crop_size_list=crop_size_list)
                    image_token = self.encode_image(image)
                image_tokens.append(image_token)
            elif m['type'] == "text":
                text += m['value']
        text = sys_part + "<user>" + text + "</user>"
        print("question:", text)
        text_token = self.tokenizer(text)['input_ids']
        image_tokens = [item for sublist in image_tokens for item in sublist]
        merge_token = text_token[:-1] + image_tokens + text_token[-1:]
        return merge_token
    
    def add_gumbel_noise(self, logits, temperature):
        '''
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        '''
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise


    def get_num_transfer_tokens(self, mask_index, steps):
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    def generate_single(self, model, x, steps=128, gen_length=128, block_length=128, temperature=0.,
                        cfg_scale=0., remasking='low_confidence', mask_id=126336, tuning=False, input_len=256):
        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, input_len + 1 + num_block * block_length: input_len + 1 + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if tuning:
                        logits = model(x_, infer=True).logits
                    else:
                        logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    if tuning:
                        logits = model(x, infer=True).logits
                    else:
                        logits = model(x).logits

                logits_with_noise = self.add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, input_len + 1 + (num_block + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index.to(x.device), x0.to(x.device), x)
                confidence = torch.where(mask_index, x0_p.to(x.device), -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    def generate_inner(self, message, dataset=None):
        full_input_token = self._prepare_content(message, dataset=dataset)
        input_len = len(full_input_token)
        total_len = input_len + 2 + self.gen_length
        token_with_mask = torch.full((1, total_len), self.mask_id, dtype=torch.long)
        token_with_mask[:,:input_len] = torch.tensor(full_input_token)
        token_with_mask[:,input_len] = 126354 # <answer> 
        token_with_mask[:,input_len+2] = 126355 # </answer>
        token_with_mask[:,input_len+3:] = 126339  # eos
        output = self.generate_single(self.model, token_with_mask, steps=self.steps,
                                     gen_length=self.gen_length, block_length=self.block_length, temperature=0., 
                                     cfg_scale=0., remasking='low_confidence', tuning=True,
                                     input_len=input_len)
        response = self.tokenizer.batch_decode(output[:,input_len+1:-1], skip_special_tokens=True)[0]
        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
