from transformers import BartModel, BartConfig
import torch.nn as nn
from .utils import load_model

class CustomBartModelWithEmbedding(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        tokenizer_src,
        tokenizer_tgt,
        share_tgt_emb_and_out=False,
        init_type="normal",
        checkpoint_inputs_embeds=None,
        checkpoint_decoder_inputs_embeds=None,
        checkpoint_bart_model=None,
        checkpoint_out=None,
    ):
        super().__init__()
        self.config = config
        
        # vocab size
        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()
        
        # Encoder Embedding
        self.inputs_embeds = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.config.d_model,
        )
        if checkpoint_inputs_embeds:
            self.inputs_embeds = load_model(
                model=self.inputs_embeds,
                checkpoint=checkpoint_inputs_embeds,
            )
        
        # Decoder Embedding
        self.decoder_inputs_embeds = nn.Embedding(
            num_embeddings=self.tgt_vocab_size,
            embedding_dim=self.config.d_model,
        )
        if checkpoint_decoder_inputs_embeds:
            self.decoder_inputs_embeds = load_model(
                model=self.decoder_inputs_embeds,
                checkpoint=checkpoint_decoder_inputs_embeds,
            )
        
        # Bart model
        self.bart_model = BartModel(config)
        if checkpoint_bart_model:
            self.bart_model = load_model(
                model=self.bart_model,
                checkpoint=checkpoint_bart_model,
            )

        # Predict
        self.out = nn.Linear(self.config.d_model, tokenizer_tgt.get_vocab_size())
        if checkpoint_out:
            self.out = load_model(
                model=self.out,
                checkpoint=checkpoint_out,
            )

        # Initialize weights
        self.initialize_weights(init_type=init_type, mean=0, std=self.config.init_std)

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
    
    def initialize_weights(self, init_type="normal", mean=0, std=0.02):
        for name, param in self.named_parameters():
            if name.startswith("bart_model"):
                continue
            if param.dim() > 1:
                if init_type == "normal":
                    nn.init.normal_(param, mean=mean, std=std)
                elif init_type == "xavier":
                    nn.init.normal_(param, mean=0, std=std)
                if name in ["inputs_embeds.weight", "decoder_inputs_embeds.weight"] and self.config.pad_token_id is not None:
                    nn.init.constant_(param[self.config.pad_token_id], 0)
                elif name in ["out.bias"]:
                    nn.init.constant_(param, 0)
                    
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