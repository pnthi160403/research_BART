from transformers import  BartConfig
import torch
from .custom_models_bart import first_train_bart_seq2seq, second_train_bart_seq2seq
from .custom_models_bart import CustomBartModel, CustomBartSeq2seq, CustomBartModelWithEmbedding
import json
from .utils import get_weights_file_path

BART = "bart"
BART_WITH_EMBEDDING = "bart_with_embedding"
BART_SEQ2SEQ = "bart_seq2seq"
STEP_TRAIN_BART_SEQ2SEQ = {
    'FIRST': first_train_bart_seq2seq,
    'SECOND': second_train_bart_seq2seq,
}

# get model config
def get_bart_config(config: dict, tokenizer_src, tokenizer_tgt):
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
        eos_token_id=tokenizer_src.token_to_id("</s>"),
        forced_bos_token_id=tokenizer_src.token_to_id("<s>"),
        forced_eos_token_id=tokenizer_src.token_to_id("</s>"),
        pad_token_id=tokenizer_src.token_to_id("<pad>"),
        num_beams=config["num_beams"],
        vocab_size=tokenizer_tgt.get_vocab_size()
    )

    if not bart_config:
        ValueError("BART config not found")

    return bart_config

# get encoder config
def get_encoder_config(config: dict, tokenizer_src, tokenizer_tgt):
    class ConfigModel():
        pass

    config_encoder = ConfigModel()
    config_encoder.d_model = config["d_model"]
    config_encoder.nhead = config["encoder_attention_heads"]
    config_encoder.vocab_size = tokenizer_src.get_vocab_size()
    config_encoder.dropout = config["dropout"]
    config_encoder.num_layers = config["encoder_layers"]
    config_encoder.encoder_ffn_dim = config["encoder_ffn_dim"]
    config_encoder.activation = config["activation_function"]

    return config_encoder

# get bart model
def get_bart_model(config: dict, tokenizer_src, tokenizer_tgt):
    bart_config = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    checkpoint = None
    if config["checkpoint_bart_model"]:
        checkpoint = config["checkpoint_bart_model"]

    model = CustomBartModel(
        config=bart_config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        checkpoint=checkpoint,
    )
    if not model:
        ValueError("Model not found")

    print("Check BART model")
    print(model)
    print("====================================================")

    return model

# get bart model seq2seq
def get_bart_model_seq2seq(config: dict, tokenizer_src, tokenizer_tgt):
    config_bart = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    config_encoder = get_encoder_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    checkpoint = config["checkpoint_bart_model"]

    model = CustomBartSeq2seq(
        config_bart=config_bart,
        config_encoder=config_encoder,
        checkpoint=checkpoint,
    )

    if not model:
        ValueError("Model not found")
    
    step_train = config["step_train"]
    if step_train:
        model = STEP_TRAIN_BART_SEQ2SEQ[step_train](
            config=config,
            model=model
        )
    
    print("Check BART model Seq2seq")
    print(model)
    print("====================================================")
    
    return model

# get bart model with embedding
def get_bart_model_with_embedding(config: dict, tokenizer_src, tokenizer_tgt):
    bart_config = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    checkpoint = config["checkpoint_bart_model"]
    share_tgt_emb_and_out = config["share_tgt_emb_and_out"]
    init_type = config["init_type"]

    model = CustomBartModelWithEmbedding(
        config=bart_config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        checkpoint=checkpoint,
        share_tgt_emb_and_out=share_tgt_emb_and_out,
        init_type=init_type,
    )

    if not model:
        ValueError("Model not found")

    print("Check BART model with embedding")
    print(model)
    print("====================================================")

    return model
    
GET_MODEL = {
    BART: get_bart_model,
    BART_WITH_EMBEDDING: get_bart_model_with_embedding,
    BART_SEQ2SEQ: get_bart_model_seq2seq,
}

# save model
def save_model(model, global_step, global_val_step, optimizer, lr_scheduler, config, save_model="model"):
    if save_model == "bart":
        model_filename = get_weights_file_path(config, f"{global_step:010d}", "bart")
    else:
        model_filename = get_weights_file_path(config, f"{global_step:010d}")

    torch.save({
        "global_step": global_step,
        "global_val_step": global_val_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict()
    }, model_filename)
    
    print(f"Saved model at {model_filename}")

# save config
def save_config(config: dict, global_step: int):
    config_filename = f"{config['model_folder']}/config_{global_step:010d}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f)
    print(f"Saved config at {config_filename}")