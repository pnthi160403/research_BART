import torch
import torch.nn as nn

def expand_mask(
    attention_mask: torch.Tensor,
):
    return attention_mask.unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3).type(torch.int64)

def causal_mask(
    tgt_len: int,
    device: torch.device,
):
    mask = torch.triu(torch.ones((1, tgt_len, tgt_len)), diagonal=1).type(torch.int64).to(device)
    return mask == 0

def create_encoder_atn_mask(
    attention_mask: torch.Tensor,
):
    return expand_mask(
        attention_mask=attention_mask,
    )

def create_decoder_atn_mask(
    attention_mask: torch.Tensor,
    tgt_len: int=None,
):
    if tgt_len is None:
        tgt_len = attention_mask.size(-1)
    causal_4d_mask = causal_mask(
        tgt_len=tgt_len,
        device=attention_mask.device,
    )
    expanded_attn_mask = expand_mask(
        attention_mask=attention_mask,
    )
    return expanded_attn_mask & causal_4d_mask

__all__ = [
    "create_encoder_atn_mask",
    "create_decoder_atn_mask",
]