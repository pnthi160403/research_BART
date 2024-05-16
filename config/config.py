import json
from pathlib import Path
import os
import glob
import torch
from ..utils import join_base

def get_config(base_dir: str=None):
    config = {}

    if not base_dir:
        config["base_dir"] = "./"
    else:
        config["base_dir"] = base_dir
    config["tokenizer_dir"] = None
    config["special_tokens"] = [
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<mask>"
    ]

    config["num_train"] = 200000
    config["num_test"] = 1000
    config["ratio_mask"] = 0.15
    config["pretrained_tokenizer"] = False
    config["model_folder"] = join_base(config["base_dir"], "/model")
    config["model_basename"] = "bart_model_"
    config["preload"] = "latest"
    config["data"] = join_base(config["base_dir"], "/data")
    config["log_dir"] = join_base(config["base_dir"], "/log")
    config["train_ds"] = None
    config["val_ds"] = None
    config["test_ds"] = None
    config["corpus"] = None
    config["vocab_size"] = 30000
    config['min_frequency'] = 2
    config["batch_train"] = 32
    config["batch_val"] = 32
    config["batch_test"] = 1
    config["epochs"] = 3
    config["max_len"] = 100

    # Pretrain
    config["pretrain"] = False

    # Tokenizer
    config["use_tokenizer"] = "wordpiece"

    # Dataset
    config["lang_src"] = "noise_vi"
    config["lang_tgt"] = "vi"

    # BART config model
    config["d_model"] = 768
    config["encoder_layes"] = 6
    config["decoder_layers"] = 6
    config["encoder_attention_heads"] = 12
    config["decoder_attention_heads"] = 12
    config["decoder_ffn_dim"] = 3072
    config["encoder_ffn_dim"] = 3072
    config["activation_function"] = "gelu"
    config["dropout"] = 0.2
    config["attention_dropout"] = 0.1
    config["activation_dropout"] = 0.1
    config["classifier_dropout"] = 0.0
    config["max_position_embeddings"] = config["max_len"] # The maximum sequence length
    config["init_std"] = 0.02 # Std for initializing all weight matrices
    config["encoder_layerdrop"] = 0.0 # Dropout encoder layer
    config["decoder_layerdrop"] = 0.0 # Dropout decoder layer
    config["scale_embedding"] = False # Scale embeddings with sqrt(d_model)
    config["num_beams"] = 4

    # Optimizer Adam
    config["weight_decay"] = 0
    config["lr"] = 0.5
    config["eps"] = 1e-9
    config["betas"] = (0.9, 0.98)
    config["label_smoothing"] = 0.01

    # Scheduler (Noam decay)
    config["warmup_steps"] = 4000

    # Different
    config["preload"] = "latest"

    # Device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Metric
    config["f_beta"] = 0.5
    config["beams"] = [2]

    return config

__all__ = [
    "get_config"
]