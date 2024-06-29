import torch
import torch.nn as nn
from .utils.init_weights import _init_weights
from .attns import (
    RELATIVE_POSITION,
)

class BartEmbeds(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        type_attn: str,
        max_position_embeddings: int=1024,
        shared: bool = False,
        embed_scale: float=1.0,
        embed_tokens: nn.Embedding=None,
        init_std: float=0.02,
    ):
        super().__init__()

        self.type_attn = type_attn
        self.embed_scale = embed_scale
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
            )
            
        if type_attn != RELATIVE_POSITION:
            self.embed_positions = nn.Embedding(
                max_position_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
            )
            self.register_buffer(
                "pos_ids",
                torch.arange(0, max_position_embeddings)
            )
            if shared:
                self.embed_positions.weight = self.embed_tokens.weight

        self.apply(lambda module: _init_weights(
            module=module,
            std=init_std,
        ))
    
    def set_embed_tokens(self, embed_tokens: nn.Embedding):
        self.embed_tokens = embed_tokens

    def forward(
            self, 
            input_ids: torch.Tensor=None,
            inputs_embeds: torch.Tensor=None,
            pos_idx: int=None,
        ):
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * self.embed_scale
        if self.type_attn != RELATIVE_POSITION:
            if input_ids is not None:
                bsz, seq_len = input_ids.size()
            else:
                bsz, seq_len, d_model = inputs_embeds.size()
            if pos_idx is not None:
                # print(f"{ pos_idx= }")
                pos_ids = self.pos_ids[pos_idx:pos_idx+1]
            else:
                pos_ids = self.pos_ids[:seq_len]
            inputs_embeds = inputs_embeds + self.embed_positions(pos_ids)
        return inputs_embeds
        
__all__ = [
    "BartEmbeds",
]