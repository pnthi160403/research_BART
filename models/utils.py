import torch
import torch.nn as nn
import os
from tqdm import tqdm

def show_layer_un_freeze(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def freeze_model(model, modules=[]):
    for module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = False
    return model

def un_freeze_model(model, modules=[]):
    for module in modules:
        for name, param in module.named_parameters():
            param.requires_grad = True
    return model

# load model state dict
def load_model(checkpoint, model):
    if torch.cuda.is_available():
        state = torch.load(checkpoint)
    else:
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state["model_state_dict"])
    return model

def calc_consine_similarity(
    E: torch.Tensor,
    vocab_size: int,
    k: int,
    eos_token_id: int,
) -> torch.Tensor:
    # E (vocab_size, d_model)
    # k: number of top cosine similarity indices to return
    top_cosine_similarity_indices = None
    for i in tqdm(range(vocab_size)):
        # (vocab_size, d_model)
        embed_i = E[i].unsqueeze(0).repeat(vocab_size, 1)
        cosine_similarities = nn.functional.cosine_similarity(
            x1=E,
            x2=embed_i,
            dim=-1,
        )
        if i != eos_token_id:
            val, idx = torch.topk(
                input=cosine_similarities,
                k=k,
            )
        else:
            val, idxs = torch.topk(
                input=cosine_similarities,
                k=k+1,
            )
            for j in range(len(idxs)):
                if idxs[j] == eos_token_id:
                    idxs = idxs.cpu().numpy().tolist()
                    idxs = idxs[:j] + idxs[j+1:]
                    val = val.cpu().numpy().tolist()
                    val = val[:j] + val[j+1:]
                    idxs = torch.tensor(idxs).to(E.device)
                    val = torch.tensor(val).to(E.device)
                    break
                if j == len(idxs) - 1:
                    idxs = idxs.cpu().numpy().tolist()
                    idxs = idxs[:k]
                    val = val.cpu().numpy().tolist()
                    val = val[:k]
                    idxs = torch.tensor(idxs).to(E.device)
                    val = torch.tensor(val).to(E.device)
        # top_cosine_similarity_indices (vocab_size, k)
        if top_cosine_similarity_indices is None:
            top_cosine_similarity_indices = idx.unsqueeze(0)
        else:
            top_cosine_similarity_indices = torch.cat([
                top_cosine_similarity_indices,
                idx.unsqueeze(0),
            ], dim=0)
    return top_cosine_similarity_indices

def get_cosine_similarity(
    path: str,
    vocab_size: int,
    k: int,
    decoder_embeds_matrix: torch.Tensor=None,
    eos_token_id: int=None,
):
    if path is not None and os.path.exists(path):
        top_cosine_similarity_indices = torch.load(path)
    else:
        top_cosine_similarity_indices = calc_consine_similarity(
            E=decoder_embeds_matrix,
            vocab_size=vocab_size,
            k=k,
            eos_token_id=eos_token_id,
        )
        if path is not None:
            torch.save(
                obj=top_cosine_similarity_indices,
                f=path,
            )
    return top_cosine_similarity_indices

__all__ = [
    "load_model",
    "freeze_model",
    "un_freeze_model",
    "show_layer_un_freeze",
    "calc_consine_similarity",
    "get_cosine_similarity",
]