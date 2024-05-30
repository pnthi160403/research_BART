from transformers import  BartConfig
import torch
from .custom_models_bart import first_fine_tune_bart_with_random_encoder, second_fine_tune_bart_with_random_encoder
from .custom_models_bart import CustomBartModel, CustomBartModelWithEmbedding, FineTuneBartWithRandomEncoder
import json
from .utils.folders import get_weights_file_path

BART = "bart"
BART_WITH_EMBEDDING = "bart_with_embedding"
FINE_TUNE_BART_WITH_RANDOM_ENCODER = "fine_tune_bart_with_random_encoder"
STEP_TRAIN_BART_SEQ2SEQ = {
    'FIRST': first_fine_tune_bart_with_random_encoder,
    'SECOND': second_fine_tune_bart_with_random_encoder,
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

    return model

# get fine tune bart model seq2seq
def get_fine_tune_bart_with_random_encoder(config: dict, tokenizer_src, tokenizer_tgt):
    bart_config = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    checkpoint = config["checkpoint"]
    if not checkpoint:
        ValueError("Checkpoint not found")

    vocab_size_encoder_bart = config["vocab_size_encoder_bart"]
    
    model = FineTuneBartWithRandomEncoder(
        config=bart_config,
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        pad_idx=tokenizer_src.token_to_id("<pad>"),
        vocab_size_encoder_bart=vocab_size_encoder_bart,
        checkpoint_custom_bart_with_embedding=checkpoint,
        init_type=config["init_type"],
    )

    if not model:
        ValueError("Model not found")
    
    step_train = config["step_train"]
    if step_train:
        model = STEP_TRAIN_BART_SEQ2SEQ[step_train](
            config=config,
            model=model
        )
    else:
        ValueError("Step train not found")

    print(model)
    
    return model

# get bart model with embedding
def get_bart_model_with_embedding(config: dict, tokenizer_src, tokenizer_tgt):
    bart_config = get_bart_config(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    share_tgt_emb_and_out = config["share_tgt_emb_and_out"]
    init_type = config["init_type"]

    model = CustomBartModelWithEmbedding(
        config=bart_config,
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        pad_idx=tokenizer_src.token_to_id("<pad>"),
        share_tgt_emb_and_out=share_tgt_emb_and_out,
        init_type=init_type,
    )

    if not model:
        ValueError("Model not found")

    return model
    
GET_MODEL = {
    BART: get_bart_model,
    BART_WITH_EMBEDDING: get_bart_model_with_embedding,
    FINE_TUNE_BART_WITH_RANDOM_ENCODER: get_fine_tune_bart_with_random_encoder,
}

# save model
def save_model(model, global_step, global_val_step, optimizer, lr_scheduler, model_folder_name, model_base_name):
    model_filename = get_weights_file_path(
        model_folder_name=model_folder_name,
        model_base_name=model_base_name,
        step=global_step
    )

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