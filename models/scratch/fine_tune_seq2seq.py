from .seq2seq import (
    BartSeq2seqConfig,
    BartSeq2seq,
)
from .utils import load_model
from transformers import BartConfig

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

class FineTuneBartSeq2seq(BartSeq2seq):
    def __init__(
        self,
        config: FineTuneBartSeq2seqConfig,
    ):
        super(FineTuneBartSeq2seq, self).__init__(
            config=config.bart_seq2seq_config
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        label=None,
    ):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            label=label,
        )
                    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        return super().get_encoder_out(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        return super().get_decoder_out(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
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