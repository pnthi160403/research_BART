import torch
import torch.nn as nn
from .architecture import (
    _init_weights,
    BartEmbeds,
)
from transformers import BartModel, BartConfig

# Class out form
class BartEncoderSeq2seqOut:
    def __init__(
        self,
        logits: torch.Tensor,
    ):
        self.last_hidden_state = logits

class BartDecoderSeq2seqOut:
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
class BartSeq2seqConfig(BartConfig):
    def __init__(
        self,
        share_tgt_emb_and_out: bool=False,
        share_vocab: bool=False,
        label_smoothing: float=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bart_config = BartConfig(**kwargs)
        self.share_tgt_emb_and_out = share_tgt_emb_and_out
        self.share_vocab = share_vocab
        self.label_smoothing = label_smoothing

# Class model
class BartSeq2seq(nn.Module):
    def __init__(
        self,
        config: BartSeq2seqConfig,
    ):
        super().__init__()

        # config
        self.config = config
        self.bart_model = BartModel(config=self.config.bart_config)
        self.out  = nn.Linear(
            config.d_model,
            self.config.vocab_size,
            bias=False,
        )

        # Initialize weights
        self.out.apply(_init_weights)

        # Share embeddings
        if config.tie_word_embeddings and config.share_vocab:
            self.bart_model._tie_weights()
        if config.share_tgt_emb_and_out:
            self.out.weight = self.bart_model.shared.weight

    def forward(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor=None,
        input_ids: torch.Tensor=None,
    ):
        outputs = self.bart_model(
            attention_mask=attention_mask,
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.config.pad_token_id,
            label_smoothing=self.config.label_smoothing,
        )
        loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return logits, loss
        
    def get_encoder_out(
        self,
        attention_mask: torch.Tensor=None,
        input_ids: torch.Tensor=None,
    ):
        outputs = self.bart_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return BartEncoderSeq2seqOut(
            logits=outputs.last_hidden_state,
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
        outputs = self.bart_model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return BartDecoderSeq2seqOut(
            logits=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
        )
    
def get_model(
    **kwargs,
):
    config = BartSeq2seqConfig(**kwargs)
    model = BartSeq2seq(
        config=config,
    )
    return model
    
__all__ = [
    "BartSeq2seq",
    "BartSeq2seqConfig",
    "get_model"
]