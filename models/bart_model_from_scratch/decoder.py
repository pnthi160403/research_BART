import torch
import torch.nn as nn
from transformers import BartConfig
from .decoder_layer import BartDecoderLayer
from .embeds import BartEmbeds
from .utils.mask import (
    create_decoder_atn_mask,
    create_encoder_atn_mask,
)

class BartDecoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layerdrop = config.decoder_layerdrop
        self.layers = nn.ModuleList([
            BartDecoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor=None,
        encoder_attention_mask: torch.Tensor=None,
        head_mask: torch.Tensor=None,
        cross_attn_head_mask: torch.Tensor=None,
    ):
        hidden_states = input_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = create_decoder_atn_mask(
                attention_mask=attention_mask,
                dtype=input_embeds.dtype,
            )
        
        if encoder_hidden_states is not None:
            encoder_attention_mask = create_encoder_atn_mask(
                attention_mask=encoder_attention_mask,
                dtype=input_embeds.dtype,
                tgt_len=hidden_states.shape[1],
            )

        # print("attention_mask shape", attention_mask.shape)
        # print("encoder_attention_mask shape", encoder_attention_mask.shape)

        for idx, decoder_layer in enumerate(self.layers):
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
            )
            hidden_states = layer_outputs

        return hidden_states
    
__all__ = [
    "BartDecoder",
]