import torch
import torch.nn as nn
from .config import BartConfig
from .encoder_layer import BartEncoderLayer

class BartEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        