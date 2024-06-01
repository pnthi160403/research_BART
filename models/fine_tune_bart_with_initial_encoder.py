from transformers import BartConfig, BartModel
from .utils import load_model, freeze_model, un_freeze_model, show_layer_un_freeze
import torch.nn as nn
from .bart_seq2seq import (
    BartSeq2seq,
    BartSeq2seqConfig,
)

class FineTuneBartWithRandomEncoderConfig:
    def __init__(
        self,
        bart_seq2seq_config,
        src_vocab_size,
        tgt_vocab_size,
        vocab_size_encoder_bart=None,
        pad_idx=None,
        init_type=None,
    ):
        self.bart_seq2seq_config = bart_seq2seq_config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.vocab_size_encoder_bart = vocab_size_encoder_bart
        self.pad_idx = pad_idx
        self.init_type = init_type
    
# Fine-tune BART with initial encoder
class FineTuneBartWithRandomEncoder(nn.Module):
    def __init__(
        self,
        config: FineTuneBartWithRandomEncoderConfig,
        checkpoint=None,
    ):
        super(FineTuneBartWithRandomEncoder, self).__init__()
        self.config = config

        # vocab size
        self.src_vocab_size = config.src_vocab_size
        self.tgt_vocab_size = config.tgt_vocab_size
        if config.vocab_size_encoder_bart is None:
            ValueError("vocab_size_encoder_bart is None")
            
        self.vocab_size_encoder_bart = config.vocab_size_encoder_bart

        # pad_idx
        self.pad_idx = config.pad_idx
        
        # Load checkpoint
        custom_bart_with_embedding = BartSeq2seq(
            config=config,
            src_vocab_size=self.vocab_size_encoder_bart,
            tgt_vocab_size=self.tgt_vocab_size,
            init_type=config.init_type,
        )
        custom_bart_with_embedding = load_model(
            model=custom_bart_with_embedding,
            checkpoint=checkpoint
        )

        # Src embedding
        self.inputs_embeds = nn.Embedding(
            num_embeddings=self.src_vocab_size,
            embedding_dim=self.config.d_model,
        )

        # Tgt embedding
        self.decoder_inputs_embeds = custom_bart_with_embedding.decoder_inputs_embeds

        # Encoder initialization
        self.random_encoder = BartModel(config).encoder

        # Pretained BART model
        self.bart_model = custom_bart_with_embedding.bart_model

        # Prediction
        self.out = custom_bart_with_embedding.out
        
        # Initialize weights xavier
        modules = [self.inputs_embeds, self.random_encoder]
        self.initialize_weights(
            modules=modules,
            init_type=config.init_type,
            mean=0,
            std=self.config.init_std
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
    
    def _init_weights(self, module, mean=0.0, std=0.02, init_type="normal"):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # for param in module.parameters():
        #     if param.dim() > 1:
        #         if init_type == "normal":
        #             nn.init.normal_(param, mean=mean, std=std)
        #         elif init_type == "xavier":
        #             nn.init.xavier_normal_(param)
        #         else:
        #             continue
    
    def initialize_weights(self, modules, init_type="normal", mean=0.0, std=0.02):
        for module in modules:
            self._init_weights(
                module=module,
                mean=mean,
                std=std,
                init_type=init_type
            )
                    
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
    freeze_modules = [
        model.bart_model,
        model.decoder_inputs_embeds,
        model.out
    ]

    model = freeze_model(
        model=model,
        modules=freeze_modules
    )

    un_freeze_modules = [
        model.bart_model.encoder.layers[0].self_attn,
        model.bart_model.encoder.embed_positions,
    ]

    model = un_freeze_model(
        model=model,
        modules=un_freeze_modules
    )

    show_layer_un_freeze(model)

    return model

def second_fine_tune_bart_with_random_encoder(config, model):
    return model

STEP_TRAIN = {
    'FIRST': first_fine_tune_bart_with_random_encoder,
    'SECOND': second_fine_tune_bart_with_random_encoder,
}

def get_model(
    bart_config,
    src_vocab_size,
    tgt_vocab_size,
    vocab_size_encoder_bart=30000,
    pad_idx=None,
    init_type=None,
    step_train=None,
    checkpoint=None,
    num_labels=None,
):
    
    bart_seq2seq_config = BartSeq2seqConfig(
        bart_config=bart_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        init_type=init_type,
    )

    config = FineTuneBartWithRandomEncoderConfig(
        bart_seq2seq_config=bart_seq2seq_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        vocab_size_encoder_bart=vocab_size_encoder_bart,
        pad_idx=pad_idx,
        init_type=init_type,
    )

    model = FineTuneBartWithRandomEncoder(
        config=config,
        checkpoint=checkpoint,
    )

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