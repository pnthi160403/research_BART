import torch
import torch.nn as nn
from transformers import BartConfig
from .encoder import BartEncoder
from .decoder import BartDecoder
from .embeds import BartEmbeds
import inspect
from .utils.init_weights import (
    _init_weights,
)
from .out_form import (
    BartEncoderOut,
    BartDecoderOut,
)       

class BartSeq2seq(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()

        # encoder_embeds
        self.inputs_embeds = BartEmbeds(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
            max_position_embeddings=config.max_position_embeddings
        )
        # decoder_embeds
        self.decoder_inputs_embeds = BartEmbeds(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
            max_position_embeddings=config.max_position_embeddings
        )
        # encoder, decoder
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)
        # out
        self.out = nn.Linear(config.d_model, config.tgt_vocab_size)
        self.apply(lambda module: _init_weights(
            module=module,
            std=config.init_std,
        ))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        # encoder
        encoder_hidden_states = self.encoder(
            input_embeds=self.inputs_embeds(input_ids),
            attention_mask=attention_mask,
        )
        # decoder
        decoder_hidden_states = self.decoder(
            input_embeds=self.decoder_inputs_embeds(decoder_input_ids),
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        # out
        out = self.out(decoder_hidden_states)
        return out
    
    def get_encoder_out(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        encoder_out = self.encoder(
            input_embeds=self.inputs_embeds(input_ids),
            attention_mask=attention_mask,
        )

        return BartEncoderOut(
            logits=encoder_out,
        )
    
    def get_decoder_out(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        decoder_out = self.decoder(
            input_embeds=self.decoder_inputs_embeds(input_ids),
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        return BartDecoderOut(
            logits=decoder_out,
        )
    
__all__ = ["BartSeq2seq"]