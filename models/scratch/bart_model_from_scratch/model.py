import torch
import torch.nn as nn
from transformers import BartConfig
from .encoder import BartEncoder
from .decoder import BartDecoder
from .utils.out_form import (
    BartEncoderOut,
    BartDecoderOut,
)

class BartModel(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()
        # pad_idx
        self.pad_idx = config.pad_idx

        # encoder, decoder
        self.encoder = BartEncoder(config)
        self.decoder = BartDecoder(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_inputs_embeds: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ):
        encoder_hidden_states = self.encoder(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        decoder_hidden_states = self.decoder(
            input_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

        hidden_states = decoder_hidden_states
        return hidden_states
    
    def get_encoder_out(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        encoder_out = self.encoder(
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return BartEncoderOut(
            logits=encoder_out,
        )
    
    def get_decoder_out(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        decoder_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        return BartDecoderOut(
            logits=decoder_out,
        )
    
__all__ = ["BartModel"]