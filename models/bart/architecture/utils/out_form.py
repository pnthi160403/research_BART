import torch

# Output Attention
class BartAttentionOut:
    def __init__(
        self,
        attn_output: torch.Tensor,
        past_key_value: list=None,
        past_attn_score: torch.Tensor=None,
        past_layer_key_value: list=None,
    ):
        self.attn_output = attn_output
        self.past_key_value = past_key_value
        self.past_attn_score = past_attn_score
        self.past_layer_key_value = past_layer_key_value

# Output Encoder
class BartEncoderOut:
    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: list=None,
        past_attn_scores: list=None,
    ):
        self.last_hidden_state = logits
        self.past_key_values = past_key_values
        self.past_attn_scores = past_attn_scores

class BartEncoderLayerOut:
    def __init__(
        self,
        out: torch.Tensor,
        present_key_value: list=None,
        present_attn_score: list=None,
    ):
        self.out = out
        self.present_key_value = present_key_value
        self.present_attn_score = present_attn_score

class BartEncoderBlockOut:
    def __init__(
        self,
        out: torch.Tensor,
        past_key_values: list=None,
        past_attn_scores: list=None,
    ):
        self.out = out
        self.past_key_values = past_key_values
        self.past_attn_scores = past_attn_scores

# Output Decoder
class BartDecoderOut:
    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: list=None,
        past_attn_scores: list=None,
    ):
        self.last_hidden_state = logits
        self.past_key_values = past_key_values
        self.past_attn_scores = past_attn_scores

class BartDecoderLayerOut:
    def __init__(
        self,
        out: torch.Tensor,
        present_key_value: list=None,
        present_attn_score: list=None,
        present_layer_key_value: list=None,
    ):
        self.out = out
        self.present_key_value = present_key_value
        self.present_attn_score = present_attn_score
        self.present_layer_key_value = present_layer_key_value

class BartDecoderBlockOut:
    def __init__(
        self,
        out: torch.Tensor,
        past_key_values: list=None,
        past_attn_scores: list=None,
    ):
        self.out = out
        self.past_key_values = past_key_values
        self.past_attn_scores = past_attn_scores

__all__ = [
    "BartAttentionOut",
    "BartEncoderOut",
    "BartEncoderLayerOut",
    "BartEncoderBlockOut",
    "BartDecoderOut",
    "BartDecoderLayerOut",
    "BartDecoderBlockOut",
]