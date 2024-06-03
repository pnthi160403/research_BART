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
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.dropout = nn.Dropout(dropout)
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
    
    @staticmethod
    def _sdpa(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, float("-inf"))
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
    ):
        query_states = self.q_proj(hidden_states)
        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else: # is cross-attention
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)

        query_states = query_states.view(query_states.shape[0], query_states.shape[1], self.num_heads,self.head_dim).transpose(1, 2)
        key_states = key_states.view(key_states.shape[0], key_states.shape[1], self.num_heads,self.head_dim).transpose(1, 2)
        value_states = value_states.view(value_states.shape[0], value_states.shape[1], self.num_heads,self.head_dim).transpose(1, 2)

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(hidden_states.shape[0], self.num_heads, hidden_states.shape[1], hidden_states.shape[1])
            attn_weights = attn_weights.view(hidden_states.shape[0] * self.num_heads, hidden_states.shape[1],hidden_states.shape[1])
            
        attn_weights = BartAttention._sdpa(query_states, key_states, value_states, attention_mask, self.dropout)
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(attn_weights.shape[0], -1, self.num_heads * self.head_dim)

        attn_output = self.out_proj(attn_weights)

        return attn_output
    
__all__ = ["BartAttention"]