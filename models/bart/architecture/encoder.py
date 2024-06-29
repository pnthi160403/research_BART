import torch
import torch.nn as nn
from .encoder_layer import BartEncoderLayer
from .config import BartConfig
from .utils import (
    BartEncoderBlockOut,
    create_encoder_atn_mask,
    expand_encoder_mask,
)
from .utils.init_weights import (
    _init_weights,
)
from .attns import (
    MULTIQUERY_SCALED_DOT_PRODUCT,
)

class BartEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        custom_encoder_layer: nn.Module=None,
    ):
        super().__init__()
        
        self.type_attn = config.type_attn
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        if custom_encoder_layer is None:
            self.layers = nn.ModuleList([
                BartEncoderLayer(config) for _ in range(config.encoder_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                custom_encoder_layer(config) for _ in range(config.encoder_layers)
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
        head_mask: torch.Tensor = None,
    ):
        hidden_states = inputs_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            input=hidden_states,
            p=self.dropout,
            training=self.training,
        )

        if attention_mask is not None:
            attention_mask = create_encoder_atn_mask(
                attention_mask=attention_mask,
            )
            attention_mask=expand_encoder_mask(
                mask=attention_mask,
                num_heads=self.num_heads,
                tgt_len=inputs_embeds.size(1),
            )

        memory_key_value_states = None
        for idx in range(len(self.layers)):
            encoder_layer = self.layers[idx]
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            encoder_layer_out_obj = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                memory_key_value_states=memory_key_value_states,
            )
            hidden_states = encoder_layer_out_obj.out
            if self.type_attn == MULTIQUERY_SCALED_DOT_PRODUCT and idx == 0:
                memory_key_value_states = encoder_layer_out_obj.present_key_value[0]

        return BartEncoderBlockOut(
            out=hidden_states,
        )
    
__all__ = [
    "BartEncoder"
]