import torch
from .utils import load_model

def first_train_bart_seq2seq(config, model):
    for param in model.bart_model.parameters():
        param.requires_grad = False
    model.bart_model.encoder.layers[0].self_attn.k_proj.weight.requires_grad = True
    model.bart_model.encoder.layers[0].self_attn.q_proj.weight.requires_grad = True
    model.bart_model.encoder.layers[0].self_attn.v_proj.weight.requires_grad = True
    model.bart_model.encoder.layers[0].self_attn.out_proj.weight.requires_grad = True
    model.bart_model.encoder.embed_positions.weight.requires_grad = True
    return model

def second_train_bart_seq2seq(config, model):
    return model

# load model state dict
def load_model(checkpoint, model):
    if torch.cuda.is_available():
        state = torch.load(checkpoint)
    else:
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state["model_state_dict"])
    return model

__all__ = ["first_train_bart_seq2seq", "second_train_bart_seq2seq", "load_model"]