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
        
    def forward(self,**kwargs):
        outputs = self.bart_model(**kwargs)
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        return logits
    
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