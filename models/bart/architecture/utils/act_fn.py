import torch.nn as nn

RELU = "relu"
GELU = "gelu"
TANH = "tanh"

ACT_FN = {
    RELU: nn.ReLU,
    GELU: nn.GELU,
    TANH: nn.Tanh,
}

__all__ = [
    "RELU",
    "GELU",
    "TANH",
    "ACT_FN",
]