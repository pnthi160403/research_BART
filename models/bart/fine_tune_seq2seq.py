from .architecture import (
    BartEmbeds,
    BartEncoder,
    BartDecoder,
    BartEncoderOut,
    BartDecoderOut,
    _init_weights,
)
from .seq2seq import (
    BartSeq2seq,
    BartSeq2seqConfig,
)
from .utils import load_model
import torch.nn as nn
import torch

# Class out form
class FineTuneEncoderBartSeq2seqOut:
    def __init__(
        self,
        logits: torch.Tensor,
    ):
        self.last_hidden_state = logits

class FineTuneDecoderBartSeq2seqOut:
    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: list=None,
        past_attn_scores: list=None,
        past_layer_key_values: list=None,
    ):
        self.last_hidden_state = logits
        self.past_key_values = past_key_values
        self.past_attn_scores = past_attn_scores
        self.past_layer_key_values = past_layer_key_values

# Class config
class FineTuneBartSeq2seqConfig(BartSeq2seqConfig):
    def __init__(
        self,
        share_tgt_emb_and_out: bool=False,
        **kwargs,
    ):
        super().__init__(
            share_tgt_emb_and_out=share_tgt_emb_and_out,
            **kwargs,
        )
        self.bart_seq2seq_config = BartSeq2seqConfig(
            share_tgt_emb_and_out=share_tgt_emb_and_out,
            **kwargs,
        )

# Class model
class FineTuneBartSeq2seq(nn.Module):
    def __init__(
        self,
        config: FineTuneBartSeq2seqConfig,
    ):
        super().__init__()

        # config
        self.config = config
        # encoder_embeds
        self.inputs_embeds = BartEmbeds(
            num_embeddings=self.config.src_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.max_position_embeddings,
            init_std=config.init_std,
            type_attn=config.type_attn,
        )
        # decoder_embeds
        self.decoder_inputs_embeds = BartEmbeds(
            num_embeddings=self.config.tgt_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.max_position_embeddings,
            init_std=config.init_std,
            type_attn=config.type_attn,
        )
        # encoder, decoder
        self.encoder = BartEncoder(config.bart_config)
        self.decoder = BartDecoder(config.bart_config)
        # out
        self.out = nn.Linear(config.bart_config.d_model, self.config.tgt_vocab_size)
        self.apply(lambda module: _init_weights(
            module=module,
            std=config.init_std,
        ))

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
        decoder_block_out_obj = self.decoder(
            inputs_embeds=self.decoder_inputs_embeds(decoder_input_ids),
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        decoder_hidden_states = decoder_block_out_obj.out
        # out
        logits = self.out(decoder_hidden_states)

        if label is not None:
            if self.config.pad_idx is not None:
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.config.pad_idx,
                    label_smoothing=self.config.label_smoothing,
                )
            else:
                loss_fn = nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing
                )
            loss = loss_fn(logits.view(-1, self.config.tgt_vocab_size), label.view(-1))
            return logits, loss
        else:
            return logits
                    
    def get_encoder_out(
        self,
        attention_mask: torch.Tensor=None,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        if inputs_embeds is not None:
            encoder_block_out_obj = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    inputs_embeds=inputs_embeds,
                ),
                attention_mask=attention_mask,
            )
        else:
            encoder_block_out_obj = self.encoder(
                inputs_embeds=self.inputs_embeds(
                    input_ids=input_ids,
                ),
                attention_mask=attention_mask,
            )
        encoder_block_out = encoder_block_out_obj.out

        return FineTuneEncoderBartSeq2seqOut(
            logits=encoder_block_out,
        )
    
    def get_decoder_out(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor=None,
        encoder_attention_mask: torch.Tensor=None,
        past_key_values: list=None,
        past_attn_scores: list=None,
        use_cache: bool=False,
        pos_idx: int=None,
    ):
        decoder_block_out_obj = self.decoder(
            inputs_embeds=self.decoder_inputs_embeds(
                input_ids=input_ids,
                use_cache=use_cache,
                pos_idx=pos_idx,
            ),
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            past_attn_scores=past_attn_scores,
            use_cache=use_cache,
        )
        decoder_block_out = decoder_block_out_obj.out
        past_key_values = decoder_block_out_obj.past_key_values
        past_attn_scores = decoder_block_out_obj.past_attn_scores

        return FineTuneDecoderBartSeq2seqOut(
            logits=decoder_block_out,
            past_key_values=past_key_values,
            past_attn_scores=past_attn_scores,
        )

def get_model(
    **kwargs,
):
    config = FineTuneBartSeq2seqConfig(
        **kwargs,
    )
    checkpoint = kwargs.get("checkpoint", None)
    # load checkpoint
    if checkpoint is None:
        ValueError("checkpoint is required")
    bart_seq2seq_model = BartSeq2seq(
        config=config.bart_seq2seq_config,
    )
    bart_seq2seq_model = load_model(
        checkpoint=checkpoint,
        model=bart_seq2seq_model,
    )
    model = FineTuneBartSeq2seq(
        config=config,
    )

    # load state dict
    model.inputs_embeds.load_state_dict(bart_seq2seq_model.inputs_embeds.state_dict())
    model.decoder_inputs_embeds.load_state_dict(bart_seq2seq_model.decoder_inputs_embeds.state_dict())
    model.encoder.load_state_dict(bart_seq2seq_model.encoder.state_dict())
    model.decoder.load_state_dict(bart_seq2seq_model.decoder.state_dict())
    model.out.load_state_dict(bart_seq2seq_model.out.state_dict())

    return model

__all__ = [
    "FineTuneBartSeq2seqConfig",
    "FineTuneBartSeq2seq",
    "get_model",
]