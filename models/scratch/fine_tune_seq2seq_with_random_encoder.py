from .utils import load_model, freeze_model, un_freeze_model, show_layer_un_freeze
import torch.nn as nn
import torch
from .bart_model_from_scratch import (
    BartEncoder,
    BartDecoder,
    BartEmbeds,
    _init_weights,
    BartEncoderOut,
    BartDecoderOut,
)
from .seq2seq import (
    BartSeq2seq,
    BartSeq2seqConfig,
)
from transformers import BartConfig

class FineTuneBartWithRandomEncoderConfig:
    def __init__(
        self,
        config_bart_seq2seq: BartSeq2seqConfig,
        config_bart: BartConfig,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx: int,
        src_vocab_size_bart_encoder: int,
        init_type: str="normal",
    ):
        self.bart_seq2seq_config = config_bart_seq2seq
        self.bart_config = config_bart
        self.pad_idx = pad_idx
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_vocab_size_bart_encoder = src_vocab_size_bart_encoder
        self.init_type = init_type

class RandomEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()
        self.inputs_embeds = BartEmbeds(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.encoder = BartEncoder(
            config=config,
        )

        # Initialize weights
        self.apply(lambda module: _init_weights(
            module=module,
            std=config.init_std,
        ))

    def forward(
        self,
        input_ids,
        attention_mask
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        inputs_embeds = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        return inputs_embeds

# Fine-tune BART with initial encoder
class FineTuneBartWithRandomEncoder(nn.Module):
    def __init__(
        self,
        config: FineTuneBartWithRandomEncoderConfig,
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

        del self.inputs_embeds.embed_tokens
        _config = config.bart_config
        _config.encoder_layers = 4
        _config.vocab_size = self.src_vocab_size
        self.random_encoder = RandomEncoder(
            config=_config
        )
        
        # Initialize weights
        _init_weights(
            module=self.out,
            std=config.bart_config.init_std,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        label=None,
    ):
        inputs_embeds = self.random_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
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
        input_ids,
        attention_mask
    ):
        inputs_embeds = self.random_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

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
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
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
    
def first_fine_tune_bart_with_random_encoder(model):
    freeze_modules = [module for module in model.modules()]
    model = freeze_model(
        model=model,
        modules=freeze_modules,
    )

    un_freeze_modules = [
        model.encoder.layers[0].self_attn.k_proj,
        model.encoder.layers[0].self_attn.v_proj,
        model.encoder.layers[0].self_attn.q_proj,
        model.encoder.layers[0].self_attn.out_proj,
        model.inputs_embeds.embed_positions,
        model.random_encoder,
    ]
    model = un_freeze_model(
        model=model,
        modules=un_freeze_modules
    )

    show_layer_un_freeze(model)

    return model

def second_fine_tune_bart_with_random_encoder(model):
    for param in model.parameters():
        if param.requires_grad == False:
            param.requires_grad = True
            
    return model

STEP_TRAIN = {
    'FIRST': first_fine_tune_bart_with_random_encoder,
    'SECOND': second_fine_tune_bart_with_random_encoder,
}

def get_model(
    bart_config,
    src_vocab_size,
    tgt_vocab_size,
    # random_encoder_layers,
    # random_decoder_layers,
    # random_encoder_attention_heads,
    # random_decoder_attention_heads,
    # random_decoder_ffn_dim,
    # random_encoder_ffn_dim,
    # random_activation_function,
    # random_dropout,
    # random_attention_dropout,
    # random_activation_dropout,
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
        src_vocab_size=src_vocab_size_bart_encoder,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        init_type=init_type,
        share_tgt_emb_and_out=share_tgt_emb_and_out,
    )

    bart_seq2seq_model = BartSeq2seq(
        config=bart_seq2seq_config,
    )

    config = FineTuneBartWithRandomEncoderConfig(
        config_bart=bart_config,
        config_bart_seq2seq=bart_seq2seq_config,
        src_vocab_size_bart_encoder=src_vocab_size_bart_encoder,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        init_type=init_type,
    )

    model = FineTuneBartWithRandomEncoder(
        config=config,
    )

    if step_train == 'FIRST':
        bart_seq2seq_model = load_model(
            model=bart_seq2seq_model,
            checkpoint=checkpoint,
        )

        model.encoder.load_state_dict(bart_seq2seq_model.encoder.state_dict())
        model.decoder.load_state_dict(bart_seq2seq_model.decoder.state_dict())
        model.decoder_inputs_embeds.load_state_dict(bart_seq2seq_model.decoder_inputs_embeds.state_dict())
        model.inputs_embeds.embed_positions.load_state_dict(bart_seq2seq_model.inputs_embeds.embed_positions.state_dict())
        model.out.load_state_dict(bart_seq2seq_model.out.state_dict())

        print("Load model from checkpoint successfully")

    if step_train:
        model = STEP_TRAIN[step_train](
            model=model
        )
    return model

__all__ = [
    "FineTuneBartWithRandomEncoder",
    "FineTuneBartWithRandomEncoderConfig",
    "first_fine_tune_bart_with_random_encoder",
    "second_fine_tune_bart_with_random_encoder",
    "get_model",
]