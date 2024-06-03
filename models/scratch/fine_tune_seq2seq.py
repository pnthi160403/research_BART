import torch
import torch.nn as nn
from transformers import BartConfig
from .bart_model_from_scratch import (
    BartEncoder,
    BartDecoder,
    BartEmbeds,
    BartEncoderOut,
    BartDecoderOut,
    _init_weights,
)
from .seq2seq import BartSeq2seq
from .utils import load_model

class FineTuneBartSeq2seq(nn.Module):
    def __init__(
        self,
        config,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        encoder=None,
        decoder=None,
        out=None,
    ):
        super().__init__()
        # pad_idx
        self.pad_idx = config.pad_token_id

        # src_vocab_size, tgt_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        self.src_vocab_size = config.src_vocab_size

        # modules to initialize
        init_modules = []

        # encoder_embeds
        if inputs_embeds is not None:
            self.inputs_embeds = inputs_embeds
        else:
            self.inputs_embeds = BartEmbeds(
                num_embeddings=self.src_vocab_size,
                embedding_dim=config.d_model,
                padding_idx=config.pad_token_id,
                max_position_embeddings=config.max_position_embeddings
            )
            init_modules.append(self.inputs_embeds)

        # decoder_embeds
        if decoder_inputs_embeds is not None:
            self.decoder_inputs_embeds = decoder_inputs_embeds
        else:
            self.decoder_inputs_embeds = BartEmbeds(
                num_embeddings=self.tgt_vocab_size,
                embedding_dim=config.d_model,
                padding_idx=config.pad_token_id,
                max_position_embeddings=config.max_position_embeddings
            )
            init_modules.append(self.decoder_inputs_embeds)

        # encoder, decoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = BartEncoder(config)
            init_modules.append(self.encoder)

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = BartDecoder(config)
            init_modules.append(self.decoder)
        
        # out
        if out is not None:
            self.out = out
        else:
            self.out = nn.Linear(config.d_model, self.tgt_vocab_size)
            init_modules.append(self.out)

        # init weights
        for module in init_modules:
            _init_weights(
                module=module,
                init_std=config.init_std,
            )

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
    pretrained_model = BartSeq2seq(
        config=config,
    )
    pretrained_model = load_model(
        checkpoint=checkpoint,
        model=pretrained_model,
    )
    
    model = FineTuneBartSeq2seq(
        config=config,
        inputs_embeds=pretrained_model.inputs_embeds,
        decoder_inputs_embeds=pretrained_model.decoder_inputs_embeds,
        encoder=pretrained_model.encoder,
        decoder=pretrained_model.decoder,
    )

    return model

__all__ = ["FineTuneBartSeq2seq", "get_model"]