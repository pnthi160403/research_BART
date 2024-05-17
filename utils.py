import torch
import numpy as np
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torchmetrics import Recall, Precision, FBetaScore, Accuracy
from torchtext.data.metrics import bleu_score
import pandas as pd
from tokenizers import Tokenizer, ByteLevelBPETokenizer

# set seed
def set_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# create dirs
def create_dirs(config: dict, dirs: list):
    created_dirs = []
    for dir_name in dirs:
        # concat with base_dir
        dir_path = config[dir_name]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        created_dirs.append(dir_path)
    
    # message
    print("Created:")
    for name_dir in created_dirs:
        print(name_dir)
    print("====================================")

# file path
def get_weights_file_path(config, epoch: str, model="model"):
    if model == "bart":
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_bart_basename']}{epoch}.pt"
    else:
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def weights_file_path(config, model="model"):
    if model == "bart":
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_bart_basename']}*"
    else:
        model_folder = f"{config['model_folder']}"
        model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files

def join_base(base_dir: str, path: str):
    return f"{base_dir}{path}"

# get optimizer lambda lr
def lambda_lr(global_step: int, config):
    global_step = max(global_step, 1)
    return (config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * config["warmup_steps"] ** (-1.5))

# figures
def draw_graph(config, title, xlabel, ylabel, data):
    x = list(range(len(data)))
    
    save_path = join_base(config['log_dir'], f"/{title}.png")
    plt.plot(x, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()
    plt.close()

# figures
def draw_multi_graph(config, title, xlabel, ylabel, all_data):
    save_path = join_base(config['log_dir'], f"/{title}.png")
    for data, info in all_data:
        x = list(range(len(data)))
        plt.plot(x, data, label=info)
        # add multiple legends
        plt.legend()

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()
    plt.close()

def figure_list_to_csv(config, column_names, data, name_csv):
    obj = {}
    for i in range(len(column_names)):
        if data[i] is not None:
            obj[str(column_names[i])] = data[i]

    data_frame = pd.DataFrame(obj, index=[0])
    save_path = join_base(config['log_dir'], f"/{name_csv}.csv")
    data_frame.to_csv(save_path, index=False)
    return data_frame

# metrics
def calc_recall(preds, target, tgt_vocab_size: int, pad_index: int, device):
    recall = Recall(task="multiclass", average='weighted', num_classes=tgt_vocab_size, ignore_index=pad_index).to(device)
    return recall(preds, target)

def calc_precision(preds, target, tgt_vocab_size: int, pad_index: int, device):
    precision = Precision(task="multiclass", average='weighted', num_classes=tgt_vocab_size, ignore_index=pad_index).to(device)
    return precision(preds, target)

def calc_accuracy(preds, target, tgt_vocab_size: int, pad_index: int, device):
    accuracy = Accuracy(task="multiclass", average='weighted', num_classes=tgt_vocab_size, ignore_index=pad_index).to(device)
    return accuracy(preds, target)

def calc_f_beta(preds, target, beta: float, tgt_vocab_size: int, pad_index: int, device):
    f_beta = FBetaScore(task="multiclass", average='weighted', num_classes=tgt_vocab_size, beta=beta, ignore_index=pad_index).to(device)
    return f_beta(preds, target)

def calc_bleu_score(refs, cands):
    scores = []
    for j in range(1, 5):
        weights = [1 / j] * j
        scores.append(bleu_score(candidate_corpus=cands,
                                 references_corpus=refs,
                                 max_n=j,
                                 weights=weights))
    return scores

# Tokenizer
# read tokenizer byte level bpe
def read_tokenizer_byte_level_bpe(config: dict):
    tokenizer_src_path = config["tokenizer_src"]
    tokenizer_tgt_path = config["tokenizer_tgt"]
    
    tokenizer_src = ByteLevelBPETokenizer.from_file(
        f"{tokenizer_src_path}/vocab.json",
        f"{tokenizer_src_path}/merges.txt"
    )
    tokenizer_tgt = ByteLevelBPETokenizer.from_file(
        f"{tokenizer_tgt_path}/vocab.json",
        f"{tokenizer_tgt_path}/merges.txt"
    )

    tokenizer_src.add_special_tokens(config["special_tokens"])

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")

    return tokenizer_src, tokenizer_tgt

# read wordpice tokenizer
def read_wordpiece_tokenizer(config: dict):
    tokenizer_src_path = config["tokenizer_src"]
    tokenizer_tgt_path = config["tokenizer_tgt"]
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")

    return tokenizer_src, tokenizer_tgt

def read_wordlevel_tokenizer(config: dict):
    tokenizer_src_path = config["tokenizer_src"]
    tokenizer_tgt_path = config["tokenizer_tgt"]
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")

    return tokenizer_src, tokenizer_tgt

def first_train_bart_seq2seq(config, model):
    for param in model.bart_model.parameters():
        param.requires_grad = False
    model.bart_model.encoder.layers[0].self_attn.k_proj.weight.requires_grad = True
    model.bart_model.encoder.layers[0].self_attn.q_proj.weight.requires_grad = True
    model.bart_model.encoder.layers[0].self_attn.v_proj.weight.requires_grad = True
    return model

def second_train_bart_seq2seq(config, model):
    return model