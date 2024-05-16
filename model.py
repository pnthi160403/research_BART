from tokenizers import ByteLevelBPETokenizer
from transformers import  BartModel, BartConfig
import torch
import torch.nn as nn
from .utils import get_weights_file_path
import json

# get model config
def get_bart_config(config: dict, tokenizer):
    # BART config
    bart_config = BartConfig(
        d_model=config["d_model"],
        encoder_layes=config["encoder_layes"],
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
        eos_token_id=tokenizer.token_to_id("</s>"),
        forced_bos_token_id=tokenizer.token_to_id("<s>"),
        forced_eos_token_id=tokenizer.token_to_id("</s>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
        num_beams=config["num_beams"],
        vocab_size=tokenizer.get_vocab_size()
    )

    if not bart_config:
        ValueError("BART config not found")

    return bart_config

# get model
def get_bart_model(config: dict, tokenizer):
    bart_config = get_bart_config(config, tokenizer)
    model = CustomBartModel(bart_config, tokenizer)
    if not model:
        ValueError("Model not found")
    return model

# save model
def save_model(model, epoch, global_step, optimizer, lr_scheduler, config, save_model="model"):
    if save_model == "bart":
        model_filename = get_weights_file_path(config, f"{epoch:02d}", "bart")
    else:
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict()
    }, model_filename)
    
    print(f"Saved model at {model_filename}")

# save config
def save_config(config: dict, epoch: int):
    config_filename = f"{config['model_folder']}/config_{epoch:02d}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f)
    print(f"Saved config at {config_filename}")

# custom BartModel
class CustomBartModel(nn.Module):
    def __init__(self, config: BartConfig, tokenizer):
        super().__init__()
        self.config = config
        self.bart_model = BartModel(config)
        self.out = nn.Linear(config.d_model, tokenizer.get_vocab_size())
        
    def forward(self,**kwargs):
        outputs = self.bart_model(**kwargs)
        last_hidden_state = outputs.last_hidden_state
        logits = self.out(last_hidden_state)
        return logits
    
    def get_encoder_out(
        self,
        input_ids,
        attention_mask
    ):
        return self.bart_model.encoder(
            input_ids,
            attention_mask
        )
    
    def get_decoder_out(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        return self.bart_model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )