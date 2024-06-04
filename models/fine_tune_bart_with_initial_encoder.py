from .transformers_huggingface import BartModel, BartEncoder, BartConfig
from .utils import load_model, freeze_model, un_freeze_model, show_layer_un_freeze
import torch.nn as nn
from .bart_seq2seq import (
    BartSeq2seq,
    BartSeq2seqConfig,
)

class FineTuneBartWithRandomEncoderConfig(BartSeq2seqConfig):
    def __init__(
        self,
        config: BartConfig,
        src_vocab_size_random_encoder: int,
    ):
        super().__init__(
            config=config,
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            pad_idx=config.pad_idx,
            share_tgt_emb_and_out=config.share_tgt_emb_and_out,
            init_type=config.init_type,
        )
        self.src_vocab_size_random_encoder = src_vocab_size_random_encoder

    
# Fine-tune BART with initial encoder
class FineTuneBartWithRandomEncoder(BartSeq2seq):
    def __init__(
        self,
        config,
        config_random_encoder,
    ):
        super(FineTuneBartWithRandomEncoder, self).__init__(config=config)

        del self.inputs_embeds
        self.inputs_embeds = nn.Embedding(
            num_embeddings=config.vocab_size_encoder_bart,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
        )
        _config = config
        _config.encoder_layers = 1
        self.random_encoder = BartEncoder(
            config=_config,
            embed_tokens=self.inputs_embeds,
        )

        # Initialize weights
        modules = [self.inputs_embeds]
        self.initialize_weights(
            init_type=config.init_type,
            modules=modules,
            mean=0,
            std=config.init_std
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        label=None,
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        inputs_embeds = self.random_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).last_hidden_state
        decoder_inputs_embeds = self.decoder_inputs_embeds(decoder_input_ids)
        outputs = self.bart_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
        )   
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)

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
                
        return logits
                    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        inputs_embeds = self.inputs_embeds(input_ids)
        inputs_embeds = self.random_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        return self.bart_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        outputs = self.bart_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        return outputs
    
def first_fine_tune_bart_with_random_encoder(config, model):
    for param in model.parameters():
        param.requires_grad = False

    un_freeze_modules = [
        model.bart_model.encoder.layers[0].self_attn.k_proj,
        model.bart_model.encoder.layers[0].self_attn.v_proj,
        model.bart_model.encoder.layers[0].self_attn.q_proj,
        model.bart_model.encoder.layers[0].self_attn.out_proj,
        model.bart_model.encoder.embed_positions,
        model.random_encoder,
    ]

    model = un_freeze_model(
        model=model,
        modules=un_freeze_modules
    )

    show_layer_un_freeze(model)

    return model

def second_fine_tune_bart_with_random_encoder(config, model):
    for param in model.parameters():
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
    pad_idx=None,
    init_type=None,
    step_train=None,
    checkpoint=None,
    num_labels=None,
    src_vocab_size_random_encoder=None,
    share_tgt_emb_and_out=False,
):
    config = bart_config
    bart_seq2seq_config = BartSeq2seqConfig(
        config=config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        init_type=init_type,
    )

    bart_seq2seq_model = BartSeq2seq(
        config=bart_seq2seq_config,
    )

    assert checkpoint, "checkpoint is required"
    bart_seq2seq_model = load_model(
        model=bart_seq2seq_model,
        checkpoint=checkpoint,
    )

    config = FineTuneBartWithRandomEncoderConfig(
        config=bart_config,
        src_vocab_size_random_encoder=src_vocab_size_random_encoder,
    )

    model = FineTuneBartWithRandomEncoder(
        config=config,
        checkpoint=checkpoint,
    )

    model.decoder_inputs_embeds.load_state_dict(bart_seq2seq_model.decoder_inputs_embeds.state_dict())
    model.out.load_state_dict(bart_seq2seq_model.out.state_dict())
    model.bart_model.load_state_dict(bart_seq2seq_model.bart_model.state_dict())

    if step_train:
        model = STEP_TRAIN[step_train](
            config=config,
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