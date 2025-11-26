import functools
import logging
import math
from typing import List
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
__all__ = ["LLaDAForMultiModalGeneration"]

def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1  # 有效位置设为1
    return attention_mask

class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)
    
    def forward(self, input_ids=None, labels=None, infer=False, **kwargs):
        if infer:
            input_ids = input_ids.tolist()
        # ========================================================
        # padding input batch len & attention bias for attention mask
        # ========================================================
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids] # every sample len --> record for attention mask
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids] # padding 0 to right --> max length
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device) 
        # attn mask
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)

        # ========================================================
        # model output 
        # ========================================================
        output = LLaDAModelLM.forward(self, input_ids=input_ids, attention_mask=attention_mask)
        if infer:
            return output

        # ========================================================
        # padding label batch len & loss
        # ========================================================
        labels = [label + [-100] * (max_tokens - len(label)) for label in labels] # padding -100 to right --> max length
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
        logits = output.logits
        loss = F.cross_entropy(logits.contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), ignore_index=-100,)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(attention_mask, labels)
            sys.exit(1)
        return loss
    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules