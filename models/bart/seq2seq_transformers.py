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

# Class model
class BartSeq2seq(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()

        # config
        self.config = config
        # encoder_embeds
        self.inputs_embeds = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        # decoder_embeds
        self.decoder_inputs_embeds = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        self.bart_model = BartModel(config=self.config)
        self.out  = nn.Linear(config.d_model, self.config.vocab_size, bias=False)

        # Share embeddings
        self.inputs_embeds.weight = self.decoder_inputs_embeds.weight
        self.out.weight = self.inputs_embeds.weight

        # Delete input embeddings
        del self.bart_model.shared
        del self.bart_model.encoder.embed_tokens
        del self.bart_model.decoder.embed_tokens

        # Initialize weights
        self.inputs_embeds.apply(_init_weights)

    def forward(
        self,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor=None,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        # encoder_head_mask (encoder_layers, encoder_attention_heads)
        # decoder_head_mask (decoder_layers, decoder_attention_heads)
        # encoder
        if input_ids is not None:
            inputs_embeds = self.inputs_embeds(input_ids)
            decoder_inputs_embeds = self.decoder_inputs_embeds(decoder_input_ids)
            outputs = self.bart_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
            )
        elif inputs_embeds is not None:
            outputs = self.bart_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
        logits = self.out(outputs.last_hidden_state)

        if labels is None:
            return (logits,)
        
        if labels is not None:
            if self.config.pad_token_id is not None:
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.config.pad_token_id,
                )
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            return logits, loss
        
    def get_encoder_out(
        self,
        attention_mask: torch.Tensor=None,
        input_ids: torch.Tensor=None,
        inputs_embeds: torch.Tensor=None,
    ):
        if input_ids is not None:
            inputs_embeds = self.inputs_embeds(input_ids)
            outputs = self.bart_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        elif inputs_embeds is not None:
            outputs = self.bart_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        logits = outputs.last_hidden_state
        return BartEncoderSeq2seqOut(
            logits=logits,
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
        inputs_embeds = self.decoder_inputs_embeds(input_ids)
        outputs = self.bart_model.decoder(
            inputs_embeds=inputs_embeds,
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
    config = BartConfig(**kwargs)
    model = BartSeq2seq(
        config=config,
    )
    return model
    
__all__ = [
    "BartSeq2seq",
    "get_model"
]