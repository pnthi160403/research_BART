from .utils import load_model, freeze_model, un_freeze_model, show_layer_un_freeze
import torch.nn as nn
import torch
from .architecture import (
    BartConfig,
    BartEncoder,
    BartDecoder,
    BartEmbeds,
    BartEncoderOut,
    BartDecoderOut,
    _init_weights,
)
from .seq2seq import (
    BartSeq2seq,
    BartSeq2seqConfig,
)

class FineTuneBartWithRandomEncoderConfig(BartSeq2seqConfig):
    def __init__(
        self,
        src_vocab_size_bart_encoder: int,
        share_tgt_emb_and_out: bool=False,
        random_encoder_layers = 4,
        random_encoder_attention_heads = 12,
        random_encoder_ffn_dim = 3072,
        random_activation_function = "gelu",
        random_dropout = 0.1,
        random_attention_dropout = 0.1,
        random_activation_dropout = 0.1,
        **kwargs,
    ):
        super().__init__(
            share_tgt_emb_and_out=share_tgt_emb_and_out,
            **kwargs,
        )
        self.src_vocab_size_bart_encoder = src_vocab_size_bart_encoder
        self.random_encoder_config = BartConfig(
            src_vocab_size=kwargs.get("src_vocab_size"),
            d_model=kwargs.get("d_model"),
            pad_idx=kwargs.get("pad_idx"),
            max_position_embeddings=kwargs.get("max_position_embeddings"),
            encoder_layers=random_encoder_layers,
            encoder_attention_heads=random_encoder_attention_heads,
            encoder_ffn_dim=random_encoder_ffn_dim,
            activation_function=random_activation_function,
            dropout=random_dropout,
            attention_dropout=random_attention_dropout,
            activation_dropout=random_activation_dropout,
            init_std=kwargs.get("init_std"),
        )
        self.bart_seq2seq_config = BartSeq2seqConfig(
            share_tgt_emb_and_out=share_tgt_emb_and_out,
            **kwargs,
        )

class RandomEncoder(nn.Module):
    def __init__(
        self,
        config: BartConfig,
    ):
        super().__init__()
        self.inputs_embeds = BartEmbeds(
            num_embeddings=config.src_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
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
    ):
        super().__init__()

        # config
        self.config = config

        # random encoder
        self.random_encoder = RandomEncoder(
            config=config.random_encoder_config,
        )
        # encoder_embeds
        self.inputs_embeds = BartEmbeds(
            num_embeddings=self.config.src_vocab_size_bart_encoder,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.max_position_embeddings,
            init_std=config.init_std,
        )
        del self.inputs_embeds.embed_tokens
        # decoder_embeds
        self.decoder_inputs_embeds = BartEmbeds(
            num_embeddings=self.config.tgt_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
            max_position_embeddings=config.max_position_embeddings,
        )
        # encoder, decoder
        self.encoder = BartEncoder(config.bart_config)
        self.decoder = BartDecoder(config.bart_config)
        # out
        self.out = nn.Linear(config.d_model, config.tgt_vocab_size)

        
        # Initialize weights
        _init_weights(
            module=self.out,
            std=config.init_std,
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
            if self.config.pad_idx is not None:
                loss_fn = nn.CrossEntropyLoss(
                    ignore_index=self.config.pad_idx,
                    label_smoothing=self.config.label_smoothing,
                )
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            loss = loss_fn(logits.view(-1, self.config.tgt_vocab_size), label.view(-1))
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
    
def first_fine_tune_bart_with_random_encoder(
    model: FineTuneBartWithRandomEncoder
):
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

def second_fine_tune_bart_with_random_encoder(
    model: FineTuneBartWithRandomEncoder
):
    for param in model.parameters():
        if param.requires_grad == False:
            param.requires_grad = True
            
    return model

STEP_TRAIN = {
    'FIRST': first_fine_tune_bart_with_random_encoder,
    'SECOND': second_fine_tune_bart_with_random_encoder,
}

def get_model(
    **kwargs,
):
    config = FineTuneBartWithRandomEncoderConfig(
        **kwargs,
    )
    bart_seq2seq_model = BartSeq2seq(
        config=config.bart_seq2seq_config,
    )
    model = FineTuneBartWithRandomEncoder(
        config=config,
    )

    step_train = kwargs.get("step_train")
    checkpoint = kwargs.get("checkpoint")
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