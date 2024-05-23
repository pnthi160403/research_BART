from transformers import BartModel, BartConfig
import torch.nn as nn
from .utils import load_model

class CustomBartModelWithEmbedding(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        tokenizer_src,
        tokenizer_tgt,
        checkpoint=None,
        share_tgt_emb_and_out=False,
    ):
        super().__init__()
        self.config = config
        
        # vocab size
        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()
        
        # Encoder Embedding
        self.inputs_embeds = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.config.d_model
        )
        
        # Decoder Embedding
        self.decoder_inputs_embeds = nn.Embedding(
            num_embeddings=self.tgt_vocab_size,
            embedding_dim=self.config.d_model
        )
        
        # Bart model
        self.bart_model = BartModel(config)
        if checkpoint:
            self.bart_model = load_model(
                model=self.bart_model,
                checkpoint=checkpoint
            )
            
        # Predict
        self.out = nn.Linear(self.config.d_model, tokenizer_tgt.get_vocab_size())

        # Initialize weights
        self.initialize_weights(mean=0, std=self.config.init_std)

        # Share the weights between embedding and linear layer
        if share_tgt_emb_and_out:
            self.out.weight = self.decoder_inputs_embeds.weight

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        decoder_inputs_embeds = self.decoder_inputs_embeds(decoder_input_ids)
        outputs = self.bart_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        return logits
    
    def initialize_weights(self, mean=0, std=0.02):
        for name, param in self.named_parameters():
            if name.startswith("bart_model"):
                continue
            if param.dim() > 1:
                nn.init.normal_(param, mean=mean, std=std)
    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        return self.bart_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        inputs_embeds = self.decoder_inputs_embeds(input_ids)
        return self.bart_model.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
    
__all__ = ["CustomBartModelWithEmbedding"]