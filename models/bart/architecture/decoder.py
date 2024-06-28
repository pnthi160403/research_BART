import torch
import torch.nn as nn
from .config import BartConfig
from .decoder_layer import BartDecoderLayer
from .utils import (
    BartDecoderBlockOut,
)
from .utils.mask import (
    create_decoder_atn_mask,
    create_encoder_atn_mask,
    expand_encoder_mask,
    expand_decoder_mask,
)

from .utils.init_weights import (
    _init_weights,
)

class BartDecoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        custom_decoder_layer: nn.Module=None,
    ):
        super().__init__()
        
        self.num_heads = config.decoder_attention_heads
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        if custom_decoder_layer is None:
            self.layers = nn.ModuleList([
                BartDecoderLayer(config) for _ in range(config.encoder_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                custom_decoder_layer(config) for _ in range(config.encoder_layers)
            ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.apply(lambda module: _init_weights(
            module=module,
            std=config.init_std,
        ))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor=None,
        encoder_attention_mask: torch.Tensor=None,
        head_mask: torch.Tensor=None,
        cross_attn_head_mask: torch.Tensor=None,
        past_key_values: list=None,
        past_attn_scores: list=None,
        use_cache: bool=False,
    ):
        hidden_states = inputs_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            input=hidden_states,
            p=self.dropout,
            training=self.training,
        )

        if attention_mask is not None:
            attention_mask = create_decoder_atn_mask(
                attention_mask=attention_mask,
                tgt_len=inputs_embeds.size(1),
            )
            attention_mask = expand_decoder_mask(
                mask=attention_mask,
                num_heads=self.num_heads,
            )
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = create_encoder_atn_mask(
                attention_mask=encoder_attention_mask,
            )
            encoder_attention_mask = expand_encoder_mask(
                mask=encoder_attention_mask,
                num_heads=self.num_heads,
                tgt_len=inputs_embeds.size(1),
            )

        next_past_key_value = [] if use_cache else None
        next_past_attn_score = [] if use_cache else None
        for idx in range(len(self.layers)):
            decoder_layer = self.layers[idx]
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            past_attn_score = past_attn_scores[idx] if past_attn_scores is not None else None
            decoder_layer_output_obj = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                past_attn_score=past_attn_score,
            )
            hidden_states = decoder_layer_output_obj.decoder_layer_out
            
            if use_cache:
                next_past_key_value.append(decoder_layer_output_obj.present_key_value)
                next_past_attn_score.append(decoder_layer_output_obj.present_attn_score)

        return BartDecoderBlockOut(
            decoder_block_out=hidden_states,
            past_key_values=next_past_key_value,
            past_attn_scores=next_past_attn_score,
        )
    
__all__ = [
    "BartDecoder",
]