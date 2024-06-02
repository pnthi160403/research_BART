import torch.nn as nn    

def _init_weights(module, mean=0.0, std=0.02):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

__all__ = [
    "_init_weights"
]