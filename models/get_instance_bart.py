from .bart_seq2seq import get_model as get_bart_seq2seq
from .fine_tune_bart_with_initial_encoder import get_model as get_fine_tune_bart_with_random_encoder
from .transformers_huggingface import BartConfig

BART_SEQ2SEQ = "bart_seq2seq"
FINE_TUNE_BART_WITH_RANDOM_ENCODER = "fine_tune_bart_with_random_encoder"

GET_MODEL = {
    BART_SEQ2SEQ: get_bart_seq2seq,
    FINE_TUNE_BART_WITH_RANDOM_ENCODER: get_fine_tune_bart_with_random_encoder,
}


# get model config
def get_bart_config(config):
    # BART config
    bart_config = BartConfig(
        d_model=config["d_model"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_attention_heads=config["encoder_attention_heads"],
        decoder_attention_heads=config["decoder_attention_heads"],
        decoder_ffn_dim=config["decoder_ffn_dim"],
        encoder_ffn_dim=config["encoder_ffn_dim"],
        activation_function=config["activation_function"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation_dropout=config["activation_dropout"],
        classifier_dropout=config["classifier_dropout"],
        max_position_embeddings=config["max_position_embeddings"],
        init_std=config["init_std"],
        encoder_layerdrop=config["encoder_layerdrop"],
        decoder_layerdrop=config["decoder_layerdrop"],
        scale_embedding=config["scale_embedding"],
        vocab_size=config["tgt_vocab_size"],
        pad_token_id=config["pad_idx"],
    )

    if not bart_config:
        ValueError("BART config not found")

    return bart_config

def get_model(config, model_train):
    src_vocab_size = config["src_vocab_size"]
    tgt_vocab_size = config["tgt_vocab_size"]
    vocab_size_encoder_bart = config["vocab_size_encoder_bart"]
    pad_idx = config["pad_idx"]
    init_type = config["init_type"]
    step_train = config["step_train"]
    checkpoint = config["checkpoint"]
    num_labels = config["num_labels"]
    share_tgt_emb_and_out = config["share_tgt_emb_and_out"]
    bart_config = get_bart_config(config)

    get_model_fn = GET_MODEL[model_train]
    model = get_model_fn(
        bart_config=bart_config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        vocab_size_encoder_bart=vocab_size_encoder_bart,
        pad_idx=pad_idx,
        init_type=init_type,
        step_train=step_train,
        num_labels=num_labels,
        checkpoint=checkpoint,
        share_tgt_emb_and_out=share_tgt_emb_and_out,
    )

    return model

__all__ = ["get_model"]