from transformers import BartModel, BartConfig
import torch.nn as nn
from .utils import load_model

class CustomBartModel(nn.Module):
    def __init__(self, config: BartConfig, tokenizer_src, tokenizer_tgt, checkpoint=None):
        super().__init__()
        self.config = config
        self.bart_model = BartModel(config)
        if checkpoint:
            self.bart_model = load_model(
                model=self.bart_model,
                checkpoint=checkpoint
            )
        self.out = nn.Linear(config.d_model, tokenizer_tgt.get_vocab_size())

        # Initialize weights
        self.out.apply(self.initialize_weights)
        
    def forward(self,**kwargs):
        outputs = self.bart_model(**kwargs)
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        return logits
    
    def initialize_weights(self, layer):
        if isinstance(layer, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
            nn.init.normal_(layer.weight, mean=0, std=self.config.init_std)
        elif isinstance(layer, nn.LayerNorm):
            layer.weight.data.fill_(1.0)
        else:
            for m in layer.modules():
                for param in m.parameters():
                    if param.dim() > 1:
                        nn.init.normal_(param, mean=0, std=self.config.init_std)
    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        return self.bart_model.encoder(
            input_ids,
            attention_mask
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        return self.bart_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
    
__all__ = ["CustomBartModel"]