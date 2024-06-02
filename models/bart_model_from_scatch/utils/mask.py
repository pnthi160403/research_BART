import torch
import torch.nn as nn

def create_encoder_atn_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int=None,
):
    bsz, src_len = attention_mask.size()
    if tgt_len is None:
        tgt_len = src_len
    expanded_attn_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_attn_mask = 1.0 - expanded_attn_mask

    return inverted_attn_mask.masked_fill(inverted_attn_mask.to(torch.bool), torch.finfo(dtype).min)

def mask_causal_mask():
    pass

def create_decoder_atn_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int=None,
):
    bsz, src_len = attention_mask.size()
    if tgt_len is None:
        tgt_len = src_len
    device = attention_mask.device
    mask = torch.full((tgt_len, tgt_len), torch.finto(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(0), 1), 0)

    mask = mask.to(dtype)
    expanded_attn_mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)
    

__all__ = ["create_encoder_atn_mask"]