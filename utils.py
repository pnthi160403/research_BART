import torch
import numpy as np
import json
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
from torchmetrics import Recall, Precision, FBetaScore, Accuracy
from torchtext.data.metrics import bleu_score
import pandas as pd

def set_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

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

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files

def lambda_lr(global_step: int, config):
    global_step = max(global_step, 1)
    return (config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * config["warmup_steps"] ** (-1.5))

def join_base(base_dir: str, path: str):
    return f"{base_dir}{path}"

import matplotlib.pyplot as plt

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

def figure_list_to_csv(config, column_names, data, name_csv):
    obj = {}
    for i in range(len(column_names)):
        if data[i] is not None:
            obj[str(column_names[i])] = data[i]

    data_frame = pd.DataFrame(obj, index=[0])
    save_path = join_base(config['log_dir'], f"/{name_csv}.csv")
    data_frame.to_csv(save_path, index=False)
    return data_frame

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