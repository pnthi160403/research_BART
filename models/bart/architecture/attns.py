import torch
import torch.nn as nn
import math
from .utils import (
    BartAttentionOut,
)

# Self-attention with scaled dot product
class MultiheadScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float=0.0,
        bias: bool=True,
        is_decoder: bool=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.is_decoder = is_decoder

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
    
    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor=None,
        dropout: nn.Dropout=None,
        past_attn_score: torch.Tensor=None,
        use_cache: bool=False,
    ) -> torch.Tensor:
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling
        if mask is not None and not use_cache:
            attention_scores.masked_fill_(mask == 0, -1e9)
            # if use_cache:
            #     # mask (batch, num_heads, q_len, k_len)
            #     attention_scores.masked_fill_(mask[:, :, -1:, :] == 0, -1e9)
            # else:
            #     attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        # if use_cache and past_attn_score is not None and attention_scores.size(-1) != past_attn_score.size(-1):
        #     past_attn_score = nn.functional.pad(
        #         input=past_attn_score,
        #         pad=(0, 1),
        #         mode='constant',
        #         value=0,
        #     )
        # if use_cache and past_attn_score is not None:
        #     attention_scores = torch.cat(
        #         [
        #             past_attn_score,
        #             attention_scores,
        #         ],
        #         dim=2,
        #     )
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # if use_cache:
        #     attn_weights = torch.matmul(attention_scores[:,:,-1:,:], value)
        # else:
        #     attn_weights = torch.matmul(attention_scores, value)
        # print(f"{ attention_scores.shape = }")
        # print(f"{ value.shape = }")
        attn_weights = torch.matmul(attention_scores, value)
        # print(f"{ attn_weights.shape = }")
        # return attn_weights[:,:,-1:,:], attention_scores
        return attn_weights, attention_scores
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        past_key_value: list=None,
        past_attn_score: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
        use_cache: bool=False,
        **kwargs,
    ):
        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, f"Hidden states have embed_dim {embed_dim}, expected {self.embed_dim}"

        is_cross_attn = key_value_states is not None
        if use_cache and is_cross_attn and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]:
            # reuse key and value in cross attention
            # query_states = self.q_proj(hidden_states[:, -1:, :])
            query_states = self.q_proj(hidden_states)
            query_states = self._shape(query_states, -1, bsz)
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attn:
            # cross attention
            query_states = self.q_proj(hidden_states)
            query_states = self._shape(query_states, -1, bsz)
            
            key_states = self.k_proj(key_value_states)
            key_states = self._shape(key_states, -1, bsz)
            
            value_states = self.v_proj(key_value_states)     
            value_states = self._shape(value_states, -1, bsz)
        elif use_cache and past_key_value is not None:
            # reuse key and value in masked self attention
            # query_states = self.q_proj(hidden_states[:, -1:, :])
            query_states = self.q_proj(hidden_states)
            query_states = self._shape(query_states, -1, bsz)

            # key_states = self.k_proj(hidden_states[:, -1:, :])
            key_states = self.k_proj(hidden_states)
            key_states = self._shape(key_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            
            # value_states = self.v_proj(hidden_states[:, -1:, :])     
            value_states = self.v_proj(hidden_states)
            value_states = self._shape(value_states, -1, bsz)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self attention
            query_states = self.q_proj(hidden_states)
            query_states = self._shape(query_states, -1, bsz)

            key_states = self.k_proj(hidden_states)
            key_states = self._shape(key_states, -1, bsz)
            
            value_states = self.v_proj(hidden_states)
            value_states = self._shape(value_states, -1, bsz)

        if self.is_decoder and use_cache:
            past_key_value = [key_states, value_states]

        attn_weights, attention_score = self.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            mask=attention_mask,
            dropout=self.dropout,
            past_attn_score=past_attn_score,
            use_cache=use_cache,
        )
        
        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights)

        past_attn_score = None
        if self.is_decoder and use_cache:
            past_attn_score = attention_score

        return BartAttentionOut(
            attn_output=attn_output,
            past_key_value=past_key_value,
            past_attn_score=past_attn_score,
        )

# Self-attention with additive attention
class MultiheadAdditiveAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float=0.0,
        bias: bool=True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.score_proj = nn.Linear(self.head_dim, 1, bias=bias)

    def _shape(
        self,
        tensor: torch.Tensor,
        seq_len: int,
        bsz: int,
    ):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def additve_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor=None,
        dropout: nn.Dropout=None,
    ) -> torch.Tensor:
        q_expand = query.unsqueeze(3).repeat_interleave(key.size(2), dim=3).contiguous()
        k_expand = key.unsqueeze(2)
        score = self.score_proj(torch.tanh(q_expand + k_expand)).squeeze(-1)
        if mask is not None:
            score = score.masked_fill_(mask == 0, -1e9)
        p_attn = nn.functional.softmax(score, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
        **kwargs,
    )-> torch.Tensor:
        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, f"Hidden states have embed_dim {embed_dim}, expected {self.embed_dim}"

        query_states = self.q_proj(hidden_states)
        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else: # is cross-attention
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)


        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        attn_weights = self.additve_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            mask=attention_mask,
            dropout=self.dropout,
        )

        if layer_head_mask is not None:
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, tgt_len)

        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights)

        return BartAttentionOut(
            attn_output=attn_output,
        )
    
# Self-attention with relative position
# Reference: https://arxiv.org/abs/1803.02155
class RelativePosition(nn.Module):
    def __init__(
        self,
        max_relative_positions: int,
        head_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_relative_positions = max_relative_positions
        self.embed_positions = nn.Parameter(torch.Tensor(max_relative_positions * 2 + 1, head_dim))
        # self.embed_positions = nn.Embedding(
        #     num_embeddings=max_relative_positions * 2 + 1,
        #     embedding_dim=head_dim,
        # )

    def forward(
        self,
        length_row: int,
        length_col: int,
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        range_row = torch.arange(length_row)
        range_col = torch.arange(length_col)
        distance = range_row[:, None] - range_col[None, :]
        distance_clip = torch.clamp(distance, -self.max_relative_positions, self.max_relative_positions)
        final_mat = torch.LongTensor(distance_clip + self.max_relative_positions).to(device)
        embeds = self.embed_positions[final_mat].to(device)
        # embeds = self.embed_positions(final_mat)
        return embeds

class MutiheadRelativeAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_relative_positions: int,
        dropout: float=0.0,
        bias: bool=True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.dropout = nn.Dropout(dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.relative_position_k = RelativePosition(
            max_relative_positions=max_relative_positions,
            head_dim=self.head_dim,
        )
        self.relative_position_v = RelativePosition(
            max_relative_positions=max_relative_positions,
            head_dim=self.head_dim,
        )

    def relative_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor=None,
        **kwargs,
    ):
        bsz = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)

        # Caculate score_edges
        # q_head (batch, num_heads, q_len, head_dim)
        q_head = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # k_head (batch, num_heads, k_len, head_dim)
        k_head = key.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # v_head (batch, num_heads, k_len, head_dim) with k_len == v_len
        v_head = value.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        # score_1 = q_head @ k_head^T
        # (batch, num_heads, q_len, head_dim) @ (batch, num_heads, head_dim, k_len)
        # -> (batch, num_heads, q_len, k_len)
        score_1 = torch.matmul(q_head, k_head.transpose(-2, -1))

        # (q_len, k_len, head_dim)
        relative_pos_k = self.relative_position_k(
            length_row=q_len,
            length_col=k_len,
        )

        # (batch, num_heads, q_len, head_dim) -> (q_len, batch * num_heads, head_dim)
        q_reshape = q_head.view(-1, q_len, self.head_dim).transpose(0, 1).contiguous()
        
        # score_2 = q_reshape @ relative_pos_k ^ T 
        # (q_len, batch * num_heads, head_dim) @ (q_len, head_dim, k_len)
        # -> (q_len, batch * num_heads, k_len)
        score_2 = torch.matmul(q_reshape, relative_pos_k.transpose(-2, -1))
        
        # (q_len, batch * num_heads, k_len) -> (batch, num_heads, q_len, k_len)
        score_2 = score_2.view(q_len, self.num_heads, -1, k_len).transpose(0, 2).contiguous()

        # (batch, num_heads, q_len, k_len)
        score_edges = ((score_1 + score_2) / self.scaling)
        if mask is not None:
            score_edges = score_edges.masked_fill_(mask == 0, -1e9)
        score_edges = self.dropout(nn.functional.softmax(
            input=score_edges,
            dim=-1,
        ))

        # Caculate weight
        # attn_weights = weight_1 + weight_2
        
        # weight_1 = score_edges @ v_head
        # (batch, num_heads, q_len, k_len) @ (batch, num_heads, k_len, head_dim)
        # -> (batch, num_heads, q_len, head_dim)
        weight_1 = torch.matmul(score_edges, v_head)

        # weight_2 = score_edges_reshape @ relative_pos_v
        # (q_len, k_len, head_dim)
        relative_pos_v = self.relative_position_v(
            length_row=q_len,
            length_col=v_len,
        )

        # (batch, num_heads, q_len, k_len) -> (q_len, batch * num_heads, k_len)
        score_edges_reshape = score_edges.view(-1, q_len, k_len).transpose(0, 1).contiguous()

        # (q_len, batch * num_heads, k_len) @ (q_len, k_len, head_dim)
        # -> (q_len, batch * num_heads, head_dim)
        weight_2 = torch.matmul(score_edges_reshape, relative_pos_v)

        # (q_len, batch * num_heads, head_dim) -> (batch, num_heads, q_len, head_dim)
        weight_2 = weight_2.view(q_len, self.num_heads, -1, self.head_dim).transpose(0, 2).contiguous()

        # (batch, num_heads, q_len, head_dim) -> (batch, q_len, num_heads * head_dim)
        return (weight_1 + weight_2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
        **kwargs,
    ):
        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, f"Hidden states have embed_dim {embed_dim}, expected {self.embed_dim}"

        query_states = self.q_proj(hidden_states)
        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else: # is cross-attention
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)

        attn_weights = self.relative_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            mask=attention_mask,
        )

        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim).contiguous()

        attn_output = self.out_proj(attn_weights)

        return BartAttentionOut(
            attn_output=attn_output,
        )
    
class MultiheadSlidingWindowSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float=0.0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.dropout = nn.Dropout(dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
        **kwargs,
    ):
        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, f"Hidden states have embed_dim {embed_dim}, expected {self.embed_dim}"

        query_states = self.q_proj(hidden_states)
        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)

        src_len = key_states.size(1)

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        attn_weights = torch.zeros(bsz, self.num_heads, tgt_len, self.head_dim).to(query_states.device)

        full_window = 2 * self.window_size + 1
        for i in range(tgt_len):
            start = max(0, i - full_window + 1)
            end = min(src_len, i + full_window)

            q_slice = query_states[:, :, i, :].unsqueeze(2)
            k_slice = key_states[:, :, start:end, :]
            v_slice = value_states[:, :, start:end, :]

            score = torch.matmul(q_slice, k_slice.transpose(-2, -1)) / self.scaling
            if attention_mask is not None:
                attn_mask_slice = attention_mask[:, :, i, start:end].unsqueeze(2)
                score.masked_fill_(attn_mask_slice == 0, -1e9)
            score = self.dropout(score)
            score = nn.functional.softmax(score, dim=-1)

            attn_weights[:, :, i, :] = torch.matmul(score, v_slice).squeeze(2)

        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights)

        return BartAttentionOut(
            attn_output=attn_output,
        )

class Tmp(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float=0.0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling = math.sqrt(self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def get_window_matrix(
        self,
        tensor: torch.Tensor,
        window_size: int,
        unfold_dim: int=2,
    ):
        pad_tensor = nn.functional.pad(
            input=tensor,
            pad=(0, 0, window_size, window_size),
            mode='constant',
            value=0,
        )
        slice_tensor = pad_tensor.unfold(
            dimension=unfold_dim,
            size=window_size * 2 + 1,
            step=1,
        ).transpose(-1, -2).contiguous()
        return slice_tensor
    
    def split_tensor(
        self,
        tensor: torch.Tensor,
        expand_len: int,
    ):
        tensor_len = tensor.size(2)
        if tensor_len > expand_len:
            if len(tensor.size()) == 5:
                tensor = tensor[:, :, :expand_len, :, :]
            elif len(tensor.size()) == 4:
                tensor = tensor[:, :, :expand_len, :]
        elif expand_len > tensor_len:
            if len(tensor.size()) == 5:
                last_element = tensor[:, :, tensor_len - 1:tensor_len, :, :]
            elif len(tensor.size()) == 4:
                last_element = tensor[:, :, tensor_len - 1:tensor_len, :]
            repeat_times = expand_len - tensor_len
            tensors_to_concat = [tensor] + [last_element] * repeat_times
            tensor = torch.cat(tensors_to_concat, dim=2)
        return tensor
    
    def split_mask(
        self,
        mask: torch.Tensor,
        window_size: int,
        expand_len: int,
    ):
        mask_expand = nn.functional.pad(
            input=mask,
            pad=(window_size, window_size),
            mode='constant',
            value=0,
        )
        mask_expand = mask_expand[:, :, :1, :].squeeze(2)
        mask_expand = mask_expand.unfold(
            dimension=2,
            size=window_size * 2 + 1,
            step=1,
        )
        mask_slice = self.split_tensor(
            tensor=mask_expand,
            expand_len=expand_len,
        )
        return mask_slice

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
        **kwargs,
    ) -> BartAttentionOut:
        bsz, tgt_len, embed_dim = hidden_states.size()
        assert embed_dim == self.embed_dim, f"Hidden states have embed_dim {embed_dim}, expected {self.embed_dim}"

        query_states = self.q_proj(hidden_states)
        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        
        self.window_size = min(self.window_size, key_states.size(1))

        query_states = query_states.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        key_slice = self.get_window_matrix(
            tensor=key_states,
            window_size=self.window_size,
        )
        key_slice = self.split_tensor(
            tensor=key_slice,
            expand_len=tgt_len,
        )
        value_slice = self.get_window_matrix(
            tensor=value_states,
            window_size=self.window_size,
        )
        value_slice = self.split_tensor(
            tensor=value_slice,
            expand_len=tgt_len,
        )
        q_expand = query_states.unsqueeze(-2)

        score = torch.matmul(q_expand, key_slice.transpose(-2, -1)).contiguous() / self.scaling
        score = score.squeeze(3)

        if attention_mask is not None:
            mask_slice = self.split_mask(
                mask=attention_mask,
                window_size=self.window_size,
                expand_len=tgt_len,
            )
            score = score.masked_fill(mask_slice == 0, -1e9)
        score = self.dropout(score)
        score = nn.functional.softmax(
            input=score,
            dim=-1,
        )
        score = score.unsqueeze(3)

        attn_weights = torch.matmul(score, value_slice).squeeze(3)
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bsz, tgt_len, self.num_heads * self.head_dim)
        
        attn_output = self.out_proj(attn_weights)

        return BartAttentionOut(
            attn_output=attn_output,
        )

# cosnt variable
SCALED_DOT_PRODUCT = "scaled_dot_product"
ADDITIVE = "additive"
RELATIVE_POSITION = "relative_position"
SLIDING_WINDOW = "sliding_window"

TYPE_ATTN = {
    SCALED_DOT_PRODUCT: MultiheadScaledDotProductAttention,
    ADDITIVE: MultiheadAdditiveAttention,
    RELATIVE_POSITION: MutiheadRelativeAttention,
    SLIDING_WINDOW: MultiheadSlidingWindowSelfAttention,
}

__all__ = [
    "MultiheadScaledDotProductAttention",
    "MultiheadAdditiveAttention",
    "MutiheadRelativeAttention",
    "SCALED_DOT_PRODUCT",
    "ADDITIVE",
    "RELATIVE_POSITION",
    "SLIDING_WINDOW",
    "TYPE_ATTN",
]