import torch
import torch.nn as nn
    
# Self-attention with scaled dot product
class MultiheadScaledDotProductAttention(nn.Module):
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
    
    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor=None,
        dropout: nn.Dropout=None,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, float("-inf"))
        p_attn = nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
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

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        attn_weights = self.scaled_dot_product_attention(
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

        return attn_output

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
            score = score.masked_fill_(mask == 0, float("-inf"))
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

        return attn_output
    
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
        self.device = kwargs.get("device", "cpu")

    def forward(
        self,
        length_row: int,
        length_col: int,
    ):
        range_row = torch.arange(length_row)
        range_col = torch.arange(length_col)
        distance = range_row[:, None] - range_col[None, :]
        distance_clip = torch.clamp(distance, -self.max_relative_positions, self.max_relative_positions)
        if torch.cuda.is_available():
            final_mat = torch.LongTensor(distance_clip + self.max_relative_positions).cuda()
            embeds = self.embed_positions[final_mat].cuda()
        else:
            final_mat = torch.LongTensor(distance_clip + self.max_relative_positions)
            embeds = self.embed_positions[final_mat]

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
        self.scaling = torch.sqrt(torch.FloatTensor([self.head_dim])).to(kwargs.get("device", "cpu"))

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
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

        q_len = query_states.size(1)
        k_len = key_states.size(1)
        v_len = value_states.size(1)

        # print(f"{ query_states.shape = }")
        # print(f"{ key_states.shape = }")
        # print(f"{ value_states.shape = }")

        # Caculate score_edges
        # q_head (batch, num_heads, q_len, head_dim)
        q_head = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # k_head (batch, num_heads, k_len, head_dim)
        k_head = key_states.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # v_head (batch, num_heads, k_len, head_dim) with k_len == v_len
        v_head = value_states.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        # score_1 = q_head @ k_head^T
        # (batch, num_heads, q_len, head_dim) @ (batch, num_heads, head_dim, k_len)
        # -> (batch, num_heads, q_len, k_len)
        score_1 = torch.matmul(q_head, k_head.transpose(-2, -1))

        # (q_len, k_len, head_dim)
        relative_pos_k = self.relative_position_k(
            length_row=q_len,
            length_col=k_len,
        )

        # print(f"{ q_head.transpose(0, 2).shape = }")
        # (batch, num_heads, q_len, head_dim) -> (q_len, batch * num_heads, head_dim)
        q_reshape = q_head.view(-1, q_len, self.head_dim).transpose(0, 1).contiguous()
        # print(f"{ q_reshape.shape = }")
        
        # score_2 = q_reshape @ relative_pos_k ^ T 
        # (q_len, batch * num_heads, head_dim) @ (q_len, head_dim, k_len)
        # -> (q_len, batch * num_heads, k_len)
        score_2 = torch.matmul(q_reshape, relative_pos_k.transpose(-2, -1))
        # print(f"{ score_2.shape = }")
        
        # (q_len, batch * num_heads, k_len) -> (batch, num_heads, q_len, k_len)
        score_2 = score_2.view(q_len, self.num_heads, -1, k_len).transpose(0, 2).contiguous()
        # print(f"{ score_2.shape = }")
        # print(f"{ score_1.shape = }")

        # (batch, num_heads, q_len, k_len)
        score_edges = (score_1 + score_2) / self.scaling
        if attention_mask is not None:
            score_edges = score_edges.masked_fill_(
                mask=attention_mask == 0,
                value=float("-inf"),
            )
        
        score_edges = self.dropout(nn.functional.softmax(
            input=score_edges,
            dim=-1,
        ))
        # print(f"{ score_edges.shape = }")

        # Caculate weight
        # attn_weights = weight_1 + weight_2
        
        # weight_1 = score_edges @ v_head
        # (batch, num_heads, q_len, k_len) @ (batch, num_heads, k_len, head_dim)
        # -> (batch, num_heads, q_len, head_dim)
        weight_1 = torch.matmul(score_edges, v_head)
        # print(f"{ weight_1.shape = }")

        # weight_2 = score_edges_reshape @ relative_pos_v
        # (q_len, k_len, head_dim)
        relative_pos_v = self.relative_position_v(
            length_row=q_len,
            length_col=v_len,
        )

        # (batch, num_heads, q_len, k_len) -> (q_len, batch * num_heads, k_len)
        score_edges_reshape = score_edges.view(-1, q_len, k_len).transpose(0, 1).contiguous()
        # print(f"{ score_edges_reshape.shape = }")
        # print(f"{ relative_pos_v.shape = }")

        # (q_len, batch * num_heads, k_len) @ (q_len, k_len, head_dim)
        # -> (q_len, batch * num_heads, head_dim)
        weight_2 = torch.matmul(score_edges_reshape, relative_pos_v)

        # (q_len, batch * num_heads, head_dim) -> (batch, num_heads, q_len, head_dim)
        weight_2 = weight_2.view(q_len, self.num_heads, -1, self.head_dim).transpose(0, 2).contiguous()
        # print(f"{ weight_2.shape = }")

        # (batch, num_heads, q_len, head_dim) -> (batch, q_len, num_heads * head_dim)
        attn_weights = (weight_1 + weight_2)
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bsz, -1, self.num_heads * self.head_dim).contiguous()

        attn_out = self.out_proj(attn_weights)
        return attn_out

# cosnt variable
SCALED_DOT_PRODUCT = "scaled_dot_product"
ADDITIVE = "additive"
RELATIVE_POSITION = "relative_position"

TYPE_ATTN = {
    SCALED_DOT_PRODUCT: MultiheadScaledDotProductAttention,
    ADDITIVE: MultiheadAdditiveAttention,
    RELATIVE_POSITION: MutiheadRelativeAttention,

}

__all__ = [
    "MultiheadScaledDotProductAttention",
    "MultiheadAdditiveAttention",
    "MutiheadRelativeAttention",
    "SCALED_DOT_PRODUCT",
    "ADDITIVE",
    "RELATIVE_POSITION",
    "TYPE_ATTN",
]