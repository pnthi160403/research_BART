import torch
from .utils.act_fn import (
    GELU,
    RELU,
    TANH,
)

class BartConfig:
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 768,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        encoder_attention_heads: int = 12,
        decoder_attention_heads: int = 12,
        decoder_ffn_dim: int = 3072,
        encoder_ffn_dim: int = 3072,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        activation_function: str = GELU,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        init_type: str = None,
        init_std: float = 0.02,
        init_mean: float = 0.0,
        label_smoothing: float = 0.01,
        pad_idx: int = 2,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.init_type = init_type
        self.init_std = init_std
        self.init_mean = init_mean
        self.label_smoothing = label_smoothing
        self.pad_idx = pad_idx

__all__ = [
    "BartConfig",
]