import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import zipfile
from torch.utils.data.distributed import DistributedSampler

def read_csv_from_zip(zip_path, csv_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(csv_filename) as csv_file:
            df = pd.read_csv(csv_file)
    return df

def get_file(file_path, file_name="zip_file.csv"):
    if zipfile.is_zipfile(file_path):
        return read_csv_from_zip(file_path, file_name)
    else:
        return pd.read_csv(file_path)

# read dataset
def read_ds(
        train_ds_path,
        val_ds_path,
        test_ds_path,
        max_num_val=10000,
        max_num_test=2000,
        max_num_train=100000,
):

    train_ds, val_ds, test_ds = None, None, None
    
    if train_ds_path and os.path.exists(train_ds_path):
        train_ds = get_file(
            file_path=train_ds_path,
            file_name="train.csv",
        )
    else:
        ValueError("Train dataset not found")

    if val_ds_path and os.path.exists(val_ds_path):
        val_ds = get_file(
            file_path=val_ds_path,
            file_name="val.csv",
        )
        if max_num_val < len(val_ds):
            val_ds = val_ds[:max_num_val]
    else:
        num_train = len(train_ds)
        num_val = min(int(num_train * 0.1), max_num_val)
        val_ds = train_ds[:num_val]
        train_ds = train_ds[num_val:]
        train_ds.reset_index(drop=True, inplace=True)
    
    if test_ds_path and os.path.exists(test_ds_path):
        test_ds = get_file(
            file_path=test_ds_path,
            file_name="test.csv",
        )
        if max_num_test < len(test_ds):
            test_ds = test_ds[:max_num_test]
    else:
        num_train = len(train_ds)
        test_ds = train_ds[:max_num_test]
        train_ds = train_ds[max_num_test:]
        train_ds.reset_index(drop=True, inplace=True)

    if len(train_ds) > max_num_train:
        train_ds = train_ds[:max_num_train]

    print("Read dataset successfully")
    print("Length train dataset: ", len(train_ds))
    print("Length val dataset: ", len(val_ds))
    print("Length test dataset: ", len(test_ds))
    print("====================================")

    return train_ds, val_ds, test_ds

# custom dataset
class Seq2seqDataset(Dataset):

    def __init__(self, ds: pd.DataFrame, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt

        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds.iloc[idx]
        src_text = src_target_pair[self.lang_src]
        tgt_text = src_target_pair[self.lang_tgt]       

        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
        }

# collate function
# define collate function
def collate_fn(batch, tokenizer_src, tokenizer_tgt):
    pad_token_id = tokenizer_src.token_to_id("<pad>")
    
    src_batch, tgt_batch, label_batch, src_text_batch, tgt_text_batch = [], [], [], [], []
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("<s>")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("</s>")], dtype=torch.int64)

    for item in batch:
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]

        enc_input_tokens = tokenizer_src.encode(src_text).ids
        dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids

        src = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
            ],
            dim=0,
        )

        tgt = torch.cat(
            [
                sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                eos_token,
            ],
            dim=0,
        )

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
def get_dataloader(
        tokenizer_src,
        tokenizer_tgt,
        batch_train,
        batch_val,
        batch_test,
        lang_src,
        lang_tgt,
        train_ds_path: str=None,
        val_ds_path: str=None,
        test_ds_path: str=None,
        max_num_val: int=15000,
        max_num_test: int=15000,
        multi_gpu: bool=False,
):
    train_ds, val_ds, test_ds = read_ds(
        train_ds_path=train_ds_path,
        val_ds_path=val_ds_path,
        test_ds_path=test_ds_path,
        max_num_val=max_num_val,
        max_num_test=max_num_test,
    )

    train_dataset, val_dataset, test_dataset = None, None, None
    train_dataloader, val_dataloader, test_dataloader = None, None, None

    train_dataset = Seq2seqDataset(
        ds=train_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
    )

    val_dataset = Seq2seqDataset(
        ds=val_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
    )

    test_dataset = Seq2seqDataset(
        ds=test_ds,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
    )

    if multi_gpu == False:
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
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_train,
            pin_memory=True,
            sampler=DistributedSampler(train_dataset),
            collate_fn=lambda batch: collate_fn(
                batch=batch,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
            ),
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

    return train_dataloader, val_dataloader, test_dataloader

__all__ = ["get_dataloader"]