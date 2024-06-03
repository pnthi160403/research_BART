import torch
import torch.nn as nn
import math

class BartAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(
        self,
        tensor: torch.Tensor,
        seq_len: int,
        bsz: int,
    ):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
    ):
        bsz, tgt_len, _ = hidden_states.size()
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)

        if key_value_states is None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        else: # is cross-attention
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)            

        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = query_states @ key_states.transpose(1, 2) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            print(f"{ attention_mask = }")
            print(f"{ attention_mask.shape = }")
            print(f"{ attn_weights.shape = }")
            attn_weights = attn_weights.masked_fill_(attention_mask == 0, -1e9)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)


        attention_mask = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
    
__all__ = ["BartAttention"]