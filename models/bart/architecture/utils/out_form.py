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
    ):
        self.last_hidden_state = logits