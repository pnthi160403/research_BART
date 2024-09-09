import torch
from transformers import Adafactor

def get_AdamW(
    model,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

def get_RAdam(
    model,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
):
    return torch.optim.RAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

def get_Adafactor(
    model,
    lr=None,
    **kwargs,
):
    return Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
    )

# const optimizer
ADAMW = "AdamW"
RADAM = "RAdam"
ADAFACTOR = "Adafactor"

GET_OPTIMIZER = {
    ADAMW: get_AdamW,
    RADAM: get_RAdam,
    ADAFACTOR: get_Adafactor,
}

__all__ = [
    "GET_OPTIMIZER",
    "ADAMW",
    "RADAM",
    "ADAFACTOR",
]