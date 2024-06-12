from .seq2seq import (
    BartSeq2seqConfig,
    BartSeq2seq,
    BartEmbeds,
    BartEncoder,
    BartDecoder,
    BartEncoderOut,
    BartDecoderOut,
    _init_weights,
)
from .utils import load_model
from transformers import BartConfig
import torch.nn as nn
import torch

class FineTuneBartSeq2seqConfig:
    def __init__(
        self,
        config_bart_seq2seq: BartSeq2seqConfig,
        config_bart: BartConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int,
        init_type: str="normal",
    ):
        self.bart_seq2seq_config = config_bart_seq2seq
        self.bart_config = config_bart
        self.pad_idx = pad_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.init_type = init_type

class FineTuneBartSeq2seq(nn.Module):
    def __init__(
        self,
        config: FineTuneBartSeq2seqConfig,
        inputs_embeds: torch.Tensor=None,
    ):
        super().__init__()
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
        if inputs_embeds is not None:
            self.inputs_embeds = inputs_embeds
        # decoder_embeds
        self.decoder_inputs_embeds = BartEmbeds(
            num_embeddings=self.tgt_vocab_size,
            embedding_dim=config.bart_config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.bart_config.max_position_embeddings,
        )
        # encoder, decoder
        self.encoder = BartEncoder(config.bart_config)
        self.decoder = BartDecoder(config.bart_config)
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
                inputs_embeds=self.inputs_embeds(
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
            inputs_embeds=self.decoder_inputs_embeds(decoder_input_ids),
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
        if inputs_embeds is not None:
            encoder_out = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    inputs_embeds=inputs_embeds,
                ),
                attention_mask=attention_mask,
            )
        else:
            encoder_out = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    input_ids=input_ids,
                ),
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
            inputs_embeds=self.decoder_inputs_embeds(input_ids),
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        return BartDecoderOut(
            logits=decoder_out,
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
    bart_seq2seq_config = BartSeq2seqConfig(
        config=bart_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        share_tgt_emb_and_out=share_tgt_emb_and_out,
        init_type=init_type,
    )

    # load checkpoint
    if checkpoint is None:
        ValueError("checkpoint is required")
    bart_seq2seq_model = BartSeq2seq(
        config=bart_seq2seq_config
    )
    bart_seq2seq_model = load_model(
        checkpoint=checkpoint,
        model=bart_seq2seq_model,
    )

    fine_tune_bart_seq2seq_config = FineTuneBartSeq2seqConfig(
        config_bart=bart_config,
        config_bart_seq2seq=bart_seq2seq_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        init_type=init_type,
    )

    model = FineTuneBartSeq2seq(
        config=fine_tune_bart_seq2seq_config,
    )

    # load state dict
    model.inputs_embeds.load_state_dict(bart_seq2seq_model.inputs_embeds.state_dict())
    model.decoder_inputs_embeds.load_state_dict(bart_seq2seq_model.decoder_inputs_embeds.state_dict())
    model.encoder.load_state_dict(bart_seq2seq_model.encoder.state_dict())
    model.decoder.load_state_dict(bart_seq2seq_model.decoder.state_dict())
    model.out.load_state_dict(bart_seq2seq_model.out.state_dict())

    return model