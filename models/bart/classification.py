import torch
import torch.nn as nn
from .architecture import (
    BartClassificationHead,
    BartEmbeds,
    BartEncoder,
    BartDecoder,
    BartConfig,
    _init_weights,
)

class BartClassificationConfig(BartConfig):
    def __init__(
        self,
        num_labels: int,
        share_tgt_emb_and_out: bool=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bart_config = BartConfig(**kwargs)
        self.num_labels = num_labels
        self.share_tgt_emb_and_out = share_tgt_emb_and_out

class BartClassification(nn.Module):
    def __init__(
        self,
        config: BartClassificationConfig,
    ):
        super().__init__()

        # config
        self.config = config
        # encoder_embeds
        self.inputs_embeds = BartEmbeds(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.max_position_embeddings,
            init_std=config.init_std,
        )
        # decoder_embeds
        self.decoder_inputs_embeds = BartEmbeds(
            num_embeddings=config.tgt_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.max_position_embeddings,
        )
        # encoder, decoder
        self.encoder = BartEncoder(config.bart_config)
        self.decoder = BartDecoder(config.bart_config)
        # out
        self.out = BartClassificationHead(
            input_dim=config.d_model,
            inner_dim=config.encoder_ffn_dim,
            num_labels=config.num_labels,
            dropout=config.dropout,
        )
        # init weights
        _init_weights(
            module=self.out,
            std=config.init_std,
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
            if self.config.pad_idx is not None:
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.config.pad_idx,
                    label_smoothing=self.config.label_smoothing,
                )
            else:
                loss_fn = nn.CrossEntropyLoss(
                    label_smoothing=self.config.label_smoothing,
                )
            loss = loss_fn(logits.view(-1, self.config.num_labels), label.view(-1))
            return logits, loss
        else:
            return logits
        
def get_model(
    **kwargs,
):
    config = BartClassificationConfig(**kwargs)
    model = BartClassification(
        config=config,
    )
    return model
        
__all__ = [
    "BartClassificationConfig",
    "BartClassification",
    "get_model",
]
