from transformers import BartModel, BartConfig
import torch.nn as nn
from .utils import load_model

class CustomBartModelWithEmbedding(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        src_vocab_size,
        tgt_vocab_size,
        pad_idx=None,
        share_tgt_emb_and_out=False,
        init_type=None,
    ):
        super().__init__()
        self.config = config
        
        # vocab size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # pad_idx
        self.pad_idx = pad_idx

        # Encoder Embedding
        self.inputs_embeds = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.config.d_model,
        )
    
        # Decoder Embedding
        self.decoder_inputs_embeds = nn.Embedding(
            num_embeddings=self.tgt_vocab_size,
            embedding_dim=self.config.d_model,
        )
    
        # Bart model
        self.bart_model = BartModel(config)

        # Prediction
        self.out = nn.Linear(self.config.d_model, self.tgt_vocab_size)
        
        # Initialize weights
        modules = [self.inputs_embeds, self.decoder_inputs_embeds, self.out]
        self.initialize_weights(
            init_type=init_type,
            modules=modules,
            mean=0,
            std=self.config.init_std
        )

        # Share the weights between embedding and linear layer
        if share_tgt_emb_and_out:
            self.out.weight = self.decoder_inputs_embeds.weight

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        label=None,
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

        if label is not None:
            if self.pad_idx is not None:
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.pad_idx,
                    label_smoothing=0.01,
                )
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
            loss = loss_fn(logits.view(-1, self.tgt_vocab_size), label.view(-1))
            return logits, loss
                    
        return logits
    
    def initialize_weights(self, modules, init_type="normal", mean=0, std=0.02):
        for module in modules:
            for param in module.parameters():
                if param.dim() > 1:
                    if init_type == "normal":
                        nn.init.normal_(param, mean=mean, std=std)
                    elif init_type == "xavier":
                        nn.init.xavier_normal_(param)
                    else:
                        continue

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