import torch
import torch.nn as nn
from .config import BartConfig
from .decoder_layer import BartDecoderLayer
from .embeds import BartEmbeds
from .utils import (
    create_encoder_atn_mask,
)

class BartEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layerdrop = config.encoder_layerdrop
        self.layers = nn.ModuleList([
            BartDecoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor = None,
    ):
        hidden_states = input_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = create_encoder_atn_mask(
                attention_mask=attention_mask,
                dtype=input_embeds.dtype,
            )

        for idx, encoder_layer in enumerate(self.layers):
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            )
            hidden_states = layer_outputs

        return hidden_states
    
__all__ = ["BartEncoder"]