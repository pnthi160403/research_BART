import torch
import torch.nn as nn
from .bart_model_from_scratch import (
    BartModel,
    BartClassificationHead,
    BartEmbeds,
    _init_weights,
)
from transformers import BartConfig

class BartClassificationConfig:
    def __init__(
        self,
        config: BartConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_labels: int,
        pad_idx: int,
        share_tgt_emb_and_out: bool=False,
        init_type: str="normal"
    ):
        self.bart_config = config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_labels = num_labels
        self.pad_idx = pad_idx
        self.share_tgt_emb_and_out = share_tgt_emb_and_out
        self.init_type = init_type


class BartClassification(BartModel):
    def __init__(
        self,
        config: BartClassificationConfig,
        num_labels: int,
        input_embeds: torch.Tensor=None,
    ):
        super().__init__(config.bart_config)
        # num_labels
        self.num_labels = num_labels
        
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
        self.out = BartClassificationHead(
            input_dim=config.bart_config.d_model,
            inner_dim=config.bart_config.encoder_ffn_dim,
            num_labels=num_labels,
            dropout=config.bart_config.dropout,
        )
        # init weights
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
        if inputs_embeds is None:
            inputs_embeds = self.inputs_embeds(input_ids)
        decoder_inputs_embeds = self.decoder_inputs_embeds(decoder_input_ids)
        hidden_states = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = self.out(
            hidden_states=hidden_states
        )

        if label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), label.view(-1))
            return logits, loss
        else:
            return logits
        
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
    config = BartClassificationConfig(
        config=bart_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_labels=num_labels,
        pad_idx=pad_idx,
        share_tgt_emb_and_out=share_tgt_emb_and_out,
        init_type=init_type,
    )
    model = BartClassification(
        config=config,
        num_labels=num_labels,
    )
    return model
        
__all__ = [
    "BartClassificationConfig",
    "BartClassification",
    "get_model",
]
