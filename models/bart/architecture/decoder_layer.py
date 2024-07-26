import torch
import torch.nn as nn
from .config import BartConfig
from .attns import (
    TYPE_ATTN,
    MULTIQUERY_SCALED_DOT_PRODUCT,
)
from .utils import (
    ACT_FN,
    GELU,
    BartDecoderLayerOut,
)

class BartDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        idx_layer = kwargs.get("idx_layer", 0)
        BartAttention = TYPE_ATTN[config.type_attn]
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            max_relative_positions=config.max_relative_positions,
            window_size=config.window_size,
            is_decoder=True,
            idx_layer=idx_layer,
        )

        self.dropout = config.dropout
        if config.activation_function != GELU:
            self.activation_fn = ACT_FN[config.activation_function]()
        else:
            self.activation_fn = ACT_FN[config.activation_function](
                approximate=config.approximate_gelu,
            )
        self.activation_dropout = nn.Dropout(config.activation_dropout)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            max_relative_positions=config.max_relative_positions,
            window_size=config.window_size,
            is_decoder=True,
            idx_layer=idx_layer,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor=None,
        encoder_hidden_states: torch.Tensor=None,
        encoder_attention_mask: torch.Tensor=None,
        layer_head_mask: torch.Tensor=None,
        cross_attn_layer_head_mask: torch.Tensor=None,
        past_key_value: list=None,
        past_attn_score: list=None,
        past_layer_key_value: list=None,
        use_cache: bool=False,
        idx_layer: int=0,
    ):
        residual = hidden_states

        present_key_value = None
        present_attn_score = None
        # Self Attention
        self_attn_past_key_value = past_key_value[0] if past_key_value is not None else None
        self_attn_past_attn_score = past_attn_score[0] if past_attn_score is not None else None
        self_attn_past_layer_key_value = past_layer_key_value[0] if past_layer_key_value is not None else None
        attn_obj = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            past_attn_score=self_attn_past_attn_score,
            past_layer_key_value=self_attn_past_layer_key_value,
            use_cache=use_cache,
            idx_layer=idx_layer,
        )
        hidden_states = attn_obj.attn_output
        present_key_value = []
        present_key_value.append(attn_obj.past_key_value)
        present_attn_score = []
        present_attn_score.append(attn_obj.past_attn_score)
        hidden_states = nn.functional.dropout(
            input=hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = hidden_states + residual
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        # Cross Attention
        if encoder_hidden_states is not None:
            cross_attn_past_key_value = past_key_value[1] if past_key_value is not None else None
            cross_attn_past_attn_score = past_attn_score[1] if past_attn_score is not None else None
            cross_attn_past_layer_key_value = past_layer_key_value[1] if past_layer_key_value is not None else None
            attn_obj = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                past_attn_score=cross_attn_past_attn_score,
                past_layer_key_value=cross_attn_past_layer_key_value,
                use_cache=use_cache,
                is_cross_attn=True,
                idx_layer=idx_layer,
            )
            hidden_states = attn_obj.attn_output
            present_key_value.append(attn_obj.past_key_value)
            present_attn_score.append(attn_obj.past_attn_score)
            hidden_states = nn.functional.dropout(
                input=hidden_states,
                p=self.dropout,
                training=self.training,
            )
            hidden_states = hidden_states + residual
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully connected layer
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)       
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            input=hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return BartDecoderLayerOut(
            out=hidden_states,
            present_key_value=present_key_value,
            present_attn_score=present_attn_score,
        )
    
__all__ = [
    "BartDecoderLayer"
]