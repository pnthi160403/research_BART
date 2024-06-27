import torch

class BartEncoderOut:
    def __init__(
        self,
        logits: torch.Tensor,
    ):
        self.last_hidden_state = logits

class BartDecoderOut:
    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: tuple=None,
    ):
        self.last_hidden_state = logits
        self.past_key_values = past_key_values

class BartAttentionOut:
    def __init__(
        self,
        attn_output: torch.Tensor,
        past_key_value: tuple=None,
    ):
        self.attn_output = attn_output
        self.past_key_value = past_key_value

class BartDecoderLayerOut:
    def __init__(
        self,
        decoder_layer_out: torch.Tensor,
        present_key_value: tuple=None,
    ):
        self.decoder_layer_out = decoder_layer_out
        self.present_key_value = present_key_value

class BartDecoderBlockOut:
    def __init__(
        self,
        decoder_block_out: torch.Tensor,
        past_key_values: tuple=None,
    ):
        self.decoder_block_out = decoder_block_out
        self.past_key_values = past_key_values

__all__ = [
    "BartEncoderOut",
    "BartDecoderOut",
    "BartAttentionOut",
    "BartDecoderLayerOut",
    "BartDecoderBlockOut",
]