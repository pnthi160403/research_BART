import torch
from tqdm import tqdm
from .beam_search import beam_search
from .utils import calc_f_beta, calc_recall, calc_precision, calc_bleu_score, set_seed
from torch.nn.utils.rnn import pad_sequence
from .prepare_dataset import read_tokenizer

def validate(model, config, beam_size, val_dataloader, num_example=5):
    set_seed()
    device = config["device"]
    
    # read tokenizer
    tokenizer = read_tokenizer(config=config)
    vocab_size=tokenizer.get_vocab_size()

    pad_token_id = tokenizer.token_to_id("<pad>")

    with torch.no_grad():

        source_texts = []
        expected = []
        predicted = []

        count = 0

        labels = []
        preds = []

        batch_iterator = tqdm(val_dataloader, desc=f"Testing model...")
        for batch in batch_iterator:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            src_text = tokenizer.decode(src[0].detach().cpu().numpy())
            tgt_text = tokenizer.decode(tgt[0].detach().cpu().numpy())

            pred_ids = beam_search(
                model=model,
                config=config,
                beam_size=beam_size,
                tokenizer=tokenizer,
                src=src_text
            )
            
            pred_text = tokenizer.decode(pred_ids.detach().cpu().numpy())
            pred_ids = torch.tensor(tokenizer.encode(pred_text).ids, dtype=torch.int64).to(device)
            label_ids = torch.tensor(tokenizer.encode(tgt_text).ids, dtype=torch.int64).to(device)
            

            padding = pad_sequence([label_ids, pred_ids], padding_value=pad_token_id, batch_first=True)
            label_ids = padding[0]
            pred_ids = padding[1]
            
            labels.append(label_ids)
            preds.append(pred_ids)

            source_texts.append(tokenizer.encode(src_text).tokens)
            expected.append([tokenizer.encode(tgt_text).tokens])
            predicted.append(tokenizer.encode(pred_text).tokens)

            count += 1

            print_step = len(val_dataloader) // num_example
            
            if count % print_step == 0:
                print()
                print(f"{f'SOURCE: ':>12}{src_text}")
                print(f"{f'TARGET: ':>12}{tgt_text}")
                print(f"{f'PREDICTED: ':>12}{pred_text}")
                print(f"{f'TOKENS TARGET: ':>12}{[tokenizer.encode(tgt_text).tokens]}")
                print(f"{f'TOKENS PREDICTED: ':>12}{tokenizer.encode(pred_text).tokens}")
                scores = calc_bleu_score(refs=[[tokenizer.encode(tgt_text).tokens]],
                                        cands=[tokenizer.encode(pred_text).tokens])
                print(f'BLEU OF SENTENCE {count}')
                for i in range(0, len(scores)):
                    print(f'BLEU_{i + 1}: {scores[i]}')
                
                print(f"{label_ids = }")
                print(f"{pred_ids = }")

                recall = calc_recall(
                    preds=pred_ids,
                    target=label_ids,
                    num_classes=vocab_size,
                    pad_index=pad_token_id,
                    device=device
                )
                precision = calc_precision(
                    preds=pred_ids,
                    target=label_ids,
                    num_classes=vocab_size,
                    pad_index=pad_token_id,
                    device=device
                )
                f_05 = calc_f_beta(
                    preds=pred_ids,
                    target=label_ids,
                    beta=config["f_beta"],
                    num_classes=vocab_size,
                    pad_index=pad_token_id,
                    device=device
                )

                recall = recall.item()
                precision = precision.item()
                f_05 = f_05.item()
                print(f"{recall = }")
                print(f"{precision = }")
                print(f"{f_05 = }")
                break

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        recall = calc_recall(
            preds=preds,
            target=labels,
            num_classes=vocab_size,
            pad_index=pad_token_id,
            device=device
        )
        precision = calc_precision(
            preds=preds,
            target=labels,
            num_classes=vocab_size,
            pad_index=pad_token_id,
            device=device
        )
        f_05 = calc_f_beta(
            preds=preds,
            target=labels,
            beta=config["f_beta"],
            num_classes=vocab_size,
            pad_index=pad_token_id,
            device=device
        )

        bleus = calc_bleu_score(refs=expected,
                                    cands=predicted)
        
        recall = recall.item()
        precision = precision.item()
        f_05 = f_05.item()
        print(f"{recall = }")
        print(f"{precision = }")
        print(f"{f_05 = }")
        
        return bleus, recall, precision, f_05