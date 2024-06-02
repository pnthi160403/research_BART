import torch
import torch.nn as nn

def expand_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int=None,
):
    bsz, src_len = attention_mask.size()
    if tgt_len is None:
        tgt_len = src_len
    expanded_attn_mask = attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    # inverted_attn_mask = 1.0 - expanded_attn_mask

    # return inverted_attn_mask.masked_fill(inverted_attn_mask.to(torch.bool), torch.finfo(dtype).min)
    return expanded_attn_mask.to(torch.int64)

def causal_mask(
    bsz: int,
    tgt_len: int,
    dtype: torch.dtype,
    device: torch.device,
):
    # mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    # mask_cond = torch.arange(mask.size(-1), device=device)
    # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    # # mask = mask.to(dtype)

    # return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)
    mask = torch.triu(torch.ones((1, tgt_len, tgt_len)), diagonal=1).type(torch.int64).to(device)
    return mask == 0

def create_encoder_atn_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int=None,
):
    return expand_mask(
        attention_mask=attention_mask,
        dtype=dtype,
        tgt_len=tgt_len,
    )

def create_decoder_atn_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int=None,
):
    if tgt_len is None:
        tgt_len = attention_mask.size(-1)
    causal_4d_mask = causal_mask(
        bsz=attention_mask.size(0),
        tgt_len=tgt_len,
        dtype=dtype,
        device=attention_mask.device,
    )
    expanded_attn_mask = expand_mask(
        attention_mask=attention_mask,
        dtype=dtype,
        tgt_len=tgt_len,
    )
    # expanded_attn_mask = causal_4d_mask.masked_fill(0, 0)
    # expanded_4d_mask = expanded_attn_mask
    return expanded_attn_mask & causal_4d_mask

__all__ = [
    "create_encoder_atn_mask",
    "create_decoder_atn_mask",
]