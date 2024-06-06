import torch
import torch.nn as nn
from transformers import BartConfig
from .bart_model_from_scratch import (
    BartEncoder,
    BartDecoder,
    BartEmbeds,
    BartEncoderOut,
    BartDecoderOut,
    BartModel,
    _init_weights,
)

class BartSeq2seqConfig:
    def __init__(
        self,
        config: BartConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int,
        share_tgt_emb_and_out: bool=False,
        init_type: str="normal",
    ):
        self.bart_config = config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = pad_idx
        self.share_tgt_emb_and_out = share_tgt_emb_and_out
        self.init_type = init_type

class BartSeq2seq(BartModel):
    def __init__(
        self,
        config: BartSeq2seqConfig,
        input_embeds: torch.Tensor=None,
    ):
        super(BartSeq2seq, self).__init__(
            config=config.bart_config,
        )
        # pad_idx
        self.pad_idx = config.pad_idx

        # src_vocab_size, tgt_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.src_vocab_size = config.src_vocab_size

        # encoder_embeds
        self.inputs_embeds = BartEmbeds(
            num_embeddings=self.src_vocab_size,
            embedding_dim=config.bart_config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.bart_config.max_position_embeddings,
            init_std=config.bart_config.init_std,
        )
        if input_embeds is not None:
            self.inputs_embeds = input_embeds
        # decoder_embeds
        self.decoder_inputs_embeds = BartEmbeds(
            num_embeddings=self.tgt_vocab_size,
            embedding_dim=config.bart_config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.bart_config.max_position_embeddings,
        )
        # out
        self.out = nn.Linear(config.bart_config.d_model, self.tgt_vocab_size)
        _init_weights(
            module=self.out,
            std=config.bart_config.init_std,
        )

    def forward(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        label: torch.Tensor=None,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        # encoder
        if inputs_embeds is not None:
            encoder_hidden_states = self.encoder(
                input_embeds=self.inputs_embeds(
                    inputs_embeds=inputs_embeds,
                ),
                attention_mask=attention_mask,
            )
        else:
            encoder_hidden_states = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    input_ids=input_ids,
                ),
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
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.inputs_embeds(input_ids)
        return super().get_encoder_out(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    
    def get_decoder_out(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        inputs_embeds = self.decoder_inputs_embeds(input_ids)
        return super().get_decoder_out(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
    
def get_model(
    bart_config,
    src_vocab_size,
    tgt_vocab_size,
    pad_idx=None,
    init_type=None,
    step_train=None,
    checkpoint=None,
    num_labels=None,
    src_vocab_size_bart_encoder=None,
    share_tgt_emb_and_out=False,
    **kwargs,
):
    config = BartSeq2seqConfig(
        config=bart_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        share_tgt_emb_and_out=share_tgt_emb_and_out,
        init_type=init_type,
    )
    model = BartSeq2seq(
        config=config,
    )

    return model
    
__all__ = ["BartSeq2seq", "get_model"]