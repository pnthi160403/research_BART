import torch
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torchmetrics import Recall, Precision, FBetaScore, Accuracy
from torchtext.data.metrics import bleu_score
import pandas as pd
from tokenizers import Tokenizer, ByteLevelBPETokenizer

from torcheval.metrics.functional.classification import multiclass_accuracy, multiclass_recall, multiclass_precision

# read and write data
def read(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            val = float(line.strip())
            data.append(val)
    return data

def write(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        for value in data:
            file.write(f"{value}\n")

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
def draw_graph(config, title, xlabel, ylabel, data, steps):
    save_path = join_base(config['log_dir'], f"/{title}.png")
    plt.plot(steps, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()
    plt.close()

# figures
def draw_multi_graph(config, title, xlabel, ylabel, all_data, steps):
    save_path = join_base(config['log_dir'], f"/{title}.png")
    for data, info in all_data:
        plt.plot(steps, data, label=info)
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

def pytorch_call_recall(input: torch.tensor, target: torch.tensor, tgt_vocab_size: int, device):
    return multiclass_recall(
        input=input,
        target=target,
        num_classes=tgt_vocab_size,
        average='weighted'
    ).to(device)

def pytorch_call_precision(input: torch.tensor, target: torch.tensor, tgt_vocab_size: int, device):
    return multiclass_precision(
        input=input,
        target=target,
        num_classes=tgt_vocab_size,
        average='weighted'
    ).to(device)

def pytorch_call_f_beta(recall: torch.tensor, precision: torch.tensor, beta: float):
    recall_item = recall.item()
    precision_item = precision.item()
    return torch.tensor((1 + beta ** 2) * (precision_item * recall_item) / ((beta ** 2 * precision_item) + recall_item), device=recall.device).type_as(recall)

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
    print(tokenizer_src_path)
    print(tokenizer_tgt_path)
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    if not tokenizer_src or not tokenizer_tgt:
        ValueError("Tokenizer not found")

    return tokenizer_src, tokenizer_tgt