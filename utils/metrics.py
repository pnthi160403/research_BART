import torch
from torcheval.metrics.functional.classification import multiclass_accuracy, multiclass_recall, multiclass_precision
# from torchtext.data.metrics import bleu_score
from torchmetrics import Recall, Precision, FBetaScore, Accuracy
from torchmetrics.text.rouge import ROUGEScore
import evaluate
rouge = evaluate.load('rouge')

# metrics
def torchmetrics_recall(preds, target, tgt_vocab_size: int, pad_index: int, device):
    recall = Recall(task="multiclass", average='weighted', num_classes=tgt_vocab_size, ignore_index=pad_index).to(device)
    return recall(preds, target)

def torchmetrics_precision(preds, target, tgt_vocab_size: int, pad_index: int, device):
    precision = Precision(task="multiclass", average='weighted', num_classes=tgt_vocab_size, ignore_index=pad_index).to(device)
    return precision(preds, target)

def torchmetrics_accuracy(preds, target, tgt_vocab_size: int, pad_index: int, device):
    accuracy = Accuracy(task="multiclass", average='weighted', num_classes=tgt_vocab_size, ignore_index=pad_index).to(device)
    return accuracy(preds, target)

def torchmetrics_f_beta(preds, target, beta: float, tgt_vocab_size: int, pad_index: int, device):
    f_beta = FBetaScore(task="multiclass", average='weighted', num_classes=tgt_vocab_size, beta=beta, ignore_index=pad_index).to(device)
    return f_beta(preds, target)

def torchmetrics_rouge(preds, target, device):
    rouge = ROUGEScore().to(device)
    return rouge(preds, target)

def torcheval_recall(input: torch.tensor, target: torch.tensor, device):
    return multiclass_recall(
        input=input,
        target=target,
    ).to(device)

def torcheval_precision(input: torch.tensor, target: torch.tensor, device):
    return multiclass_precision(
        input=input,
        target=target,
    ).to(device)

def torcheval_f_beta(recall: torch.tensor, precision: torch.tensor, beta: float):
    recall_item = recall.item()
    precision_item = precision.item()
    esi = 1e-7

    return torch.tensor((1 + beta ** 2) * (precision_item * recall_item) / (beta ** 2 * precision_item + recall_item + esi), dtype=recall.dtype).to(recall.device)

def compute_rouges(preds, refs, device):
    preds = preds.to(device)
    refs = refs.to(device)
    rouge = evaluate.load('rouge')
    res = rouge.compute(predictions=preds, references=refs)
    ans = {
        "rouge1_fmeasure": res['rouge1'],
        "rouge2_fmeasure": res['rouge2'],
        "rougeL_fmeasure": res['rougeL'],
    }
    return ans

__all__ = [
    "torchmetrics_recall",
    "torchmetrics_precision",
    "torchmetrics_accuracy",
    "torchmetrics_f_beta",
    "torchmetrics_rouge",
    "torcheval_recall",
    "torcheval_precision",
    "torcheval_f_beta",
]