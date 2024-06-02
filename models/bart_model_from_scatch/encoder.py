import torch
import torch.nn as nn
from .config import BartConfig
from .encoder_layer import BartEncoderLayer
from .embeds import BartEmbeds

class BartEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(config.dropout)
        self.layerdrop = nn.Dropout(config.encoder_layerdrop)
        self.layers = nn.ModuleList([
            BartEncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor = None,
    ):
        hidden_states = input_embeds
        hiedden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            