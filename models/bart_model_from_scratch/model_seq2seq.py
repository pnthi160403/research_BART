import torch
import torch.nn as nn
from transformers import BartConfig
from .encoder import BartEncoder
from .decoder import BartDecoder
from .embeds import BartEmbeds
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
        config,
    ):
        super().__init__()
        # pad_idx
        self.pad_idx = config.pad_token_id

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
        label: torch.Tensor=None,
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
        logits = self.out(decoder_hidden_states)

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
        else:
            return logits
    
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
    
def get_model(
    bart_config: BartConfig,
    src_vocab_size: int,
    tgt_vocab_size: int,
    vocab_size_encoder_bart: int=None,
    pad_idx: int=2,
    init_type: str="normal",
    step_train: str=None,
    num_labels: int=None,
    checkpoint: str=None,
    share_tgt_emb_and_out: bool=False,
):
    config = bart_config
    config.src_vocab_size = src_vocab_size
    config.tgt_vocab_size = tgt_vocab_size
    config.pad_token_id = pad_idx

    model = BartSeq2seq(
        config=config,
    )
    
__all__ = ["BartSeq2seq", "get_model"]