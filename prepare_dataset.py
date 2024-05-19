import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os

from .utils import read_tokenizer_byte_level_bpe, read_wordpiece_tokenizer, read_wordlevel_tokenizer

# read dataset
def read_ds(config: dict):
    train_ds_path = config["train_ds"]
    val_ds_path = config["val_ds"]
    test_ds_path = config["test_ds"]

    train_ds, val_ds, test_ds = None, None, None
    
    if train_ds_path and os.path.exists(train_ds_path):
        train_ds = pd.read_csv(train_ds_path)
    else:
        ValueError("Train dataset not found")

    if val_ds_path and os.path.exists(val_ds_path):
        val_ds = pd.read_csv(val_ds_path)
    else:
        num_train = len(train_ds)
        num_val = int(num_train * 0.1)
        val_ds = train_ds[:num_val]
        train_ds = train_ds[num_val:]
        train_ds.reset_index(drop=True, inplace=True)
    
    if test_ds_path and os.path.exists(test_ds_path):
        test_ds = pd.read_csv(test_ds_path)
    else:
        num_train = len(train_ds)
        num_test = 3000
        test_ds = train_ds[:num_test]
        train_ds = train_ds[num_test:]
        train_ds.reset_index(drop=True, inplace=True)

    print("Read dataset successfully")
    print("Train dataset")
    print(train_ds.head())

    print("Val dataset")
    print(val_ds.head())

    print("Test dataset")
    print(test_ds.head())
    print("====================================")

    return train_ds, val_ds, test_ds

def read_tokenizer(config: dict):
    if config["use_tokenizer"] == "byte-level-bpe":
        tokenizer_src, tokenizer_tgt = read_tokenizer_byte_level_bpe(config)
    if config["use_tokenizer"] == "wordpiece":
        tokenizer_src, tokenizer_tgt = read_wordpiece_tokenizer(config)
    if config["use_tokenizer"] == "wordlevel":
        tokenizer_src, tokenizer_tgt = read_wordlevel_tokenizer(config)
    print("Read tokenizer successfully")
    print("Check tokenizer src")
    print(tokenizer_src.get_vocab_size())
    print("Check tokenizer tgt")
    print(tokenizer_tgt.get_vocab_size())
    print("====================================")
    
    return tokenizer_src, tokenizer_tgt

# custom dataset
class CustomDataset(Dataset):

    def __init__(self, ds: pd.DataFrame, tokenizer_src, tokenizer_tgt, config: dict):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = config["lang_src"]
        self.tgt_lang = config["lang_tgt"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds.iloc[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]       
        sos_token_id = self.tokenizer_tgt.token_to_id("<s>")
        eos_token_id = self.tokenizer_tgt.token_to_id("</s>")

        src = [sos_token_id] + self.tokenizer_src.encode(src_text).ids + [eos_token_id]
        tgt = [sos_token_id] + self.tokenizer_tgt.encode(tgt_text).ids
        label = self.tokenizer_tgt.encode(tgt_text).ids + [eos_token_id]

        return {
            'src': src, # <s>...</s>
            "tgt": tgt, # <s> ...
            'label': label, # ...</s>
            'src_text': src_text,
            'tgt_text': tgt_text,
        }

# collate function
# define collate function
def collate_fn(batch, tokenizer_src, tokenizer_tgt):
    pad_token_id = tokenizer_src.token_to_id("<pad>")
    
    src_batch, tgt_batch, label_batch, src_text_batch, tgt_text_batch = [], [], [], [], []
    for item in batch:
        src = torch.tensor(item["src"], dtype=torch.int64)
        tgt = torch.tensor(item['tgt'], dtype=torch.int64)
        label = torch.tensor(item['label'], dtype=torch.int64)
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]
        
        src_batch.append(src)
        tgt_batch.append(tgt)
        label_batch.append(label)
        src_text_batch.append(src_text)
        tgt_text_batch.append(tgt_text)
        
    src_batch = pad_sequence(src_batch, padding_value=pad_token_id, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_token_id, batch_first=True)
    
    label_batch = pad_sequence(label_batch, padding_value=pad_token_id, batch_first=True)
    
    return {
        'src': src_batch,
        "tgt": tgt_batch,
        'label': label_batch,
        'src_text': src_text_batch,
        'tgt_text': tgt_text_batch,
    }

# get dataloader dataset
def get_dataloader(config: dict):
    tokenizer_src, tokenizer_tgt = read_tokenizer(config)
    train_ds, val_ds, test_ds = read_ds(config)

    batch_train = config['batch_train']
    batch_val = config['batch_val']
    batch_test = config['batch_test']

    train_dataset, val_dataset, test_dataset = None, None, None
    train_dataloader, val_dataloader, test_dataloader = None, None, None

    train_dataset = CustomDataset(
        ds=train_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        config=config
    )

    val_dataset = CustomDataset(
        ds=val_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        config=config
    )

    test_dataset = CustomDataset(
        ds=test_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        config=config
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_val,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_test,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
        )
    )

    ValueError("Dataloader not found")

    print("Get dataloader successfully")
    print("Train dataloader")
    print(train_dataloader)

    print("Val dataloader")
    print(val_dataloader)
    
    print("Test dataloader")
    print(test_dataloader)

    return train_dataloader, val_dataloader, test_dataloader