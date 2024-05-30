import torch
from tqdm import tqdm
from .beam_search import beam_search
from torch.nn.utils.rnn import pad_sequence
# import evaluate
from .utils.tokenizers import read_tokenizer
from .utils.metrics import (
    torchmetrics_accuracy,
    torchmetrics_recall,
    torchmetrics_precision,
    torchmetrics_f_beta,
    torcheval_recall,
    torcheval_precision,
    torcheval_f_beta,
    torchtext_bleu_score
)

def validate(model, config, beam_size, val_dataloader, num_example=20):
    device = config["device"]
    
    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )
        
    vocab_size=tokenizer_tgt.get_vocab_size()
    pad_token_id = tokenizer_src.token_to_id("<pad>")

    with torch.no_grad():

        source_texts = []
        expected = []
        predicted = []

        count = 0

        labels = []
        preds = []

        batch_iterator = tqdm(val_dataloader, desc=f"Testing model...")
        for batch in batch_iterator:
            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]

            pred_ids = beam_search(
                model=model,
                config=config,
                beam_size=beam_size,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                src=src_text
            )
            
            pred_text = tokenizer_tgt.decode(pred_ids.detach().cpu().numpy())
            pred_ids = torch.tensor(tokenizer_tgt.encode(pred_text).ids, dtype=torch.int64).to(device)
            label_ids = torch.tensor(tokenizer_tgt.encode(tgt_text).ids, dtype=torch.int64).to(device)

            padding = pad_sequence([label_ids, pred_ids], padding_value=pad_token_id, batch_first=True)
            label_ids = padding[0]
            pred_ids = padding[1]
            
            labels.append(label_ids)
            preds.append(pred_ids)

            source_texts.append(tokenizer_src.encode(src_text).tokens)
            expected.append([tokenizer_tgt.encode(tgt_text).tokens])
            predicted.append(tokenizer_tgt.encode(pred_text).tokens)

            # debug
            break

            count += 1

            print_step = max(len(val_dataloader) // num_example, 1)
            
            if count % print_step == 0:
                print()
                print(f"{f'SOURCE: ':>12}{src_text}")
                print(f"{f'TARGET: ':>12}{tgt_text}")
                print(f"{f'PREDICTED: ':>12}{pred_text}")
                print(f"{f'TOKENS TARGET: ':>12}{[tokenizer_tgt.encode(tgt_text).tokens]}")
                print(f"{f'TOKENS PREDICTED: ':>12}{tokenizer_tgt.encode(pred_text).tokens}")
                scores = torchtext_bleu_score(refs=[[tokenizer_tgt.encode(tgt_text).tokens]],
                                        cands=[tokenizer_tgt.encode(pred_text).tokens])
                print(f'BLEU OF SENTENCE {count}')
                for i in range(0, len(scores)):
                    print(f'BLEU_{i + 1}: {scores[i]}')
                
                if not config["use_pytorch_metric"]:
                    recall = torchmetrics_recall(
                        preds=pred_ids,
                        target=label_ids,
                        tgt_vocab_size=vocab_size,
                        pad_index=pad_token_id,
                        device=device
                    )
                    precision = torchmetrics_precision(
                        preds=pred_ids,
                        target=label_ids,
                        tgt_vocab_size=vocab_size,
                        pad_index=pad_token_id,
                        device=device
                    )
                    # f_05 = torchmetrics_f_beta(
                    #     preds=pred_ids,
                    #     target=label_ids,
                    #     beta=config["f_beta"],
                    #     tgt_vocab_size=vocab_size,
                    #     pad_index=pad_token_id,
                    #     device=device
                    # )
                else:
                    recall = torcheval_recall(
                        input=pred_ids,
                        target=label_ids,
                        device=device
                    )

                    precision = torcheval_precision(
                        input=pred_ids,
                        target=label_ids,
                        device=device
                    )

                    # f_05 = torcheval_f_beta(
                    #     recall=recall,
                    #     precision=precision,
                    #     beta=config["f_beta"]
                    # )

                recall = recall.item()
                precision = precision.item()
                # f_05 = f_05.item()
                print(f"{recall = }")
                print(f"{precision = }")
                # print(f"{f_05 = }")
            
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        if not config["use_pytorch_metric"]:
            recall = torchmetrics_recall(
                preds=preds,
                target=labels,
                tgt_vocab_size=vocab_size,
                pad_index=pad_token_id,
                device=device
                )
            precision = torchmetrics_precision(
                preds=preds,
                target=labels,
                tgt_vocab_size=vocab_size,
                pad_index=pad_token_id,
                device=device
                )
            # f_05 = torchmetrics_f_beta(
            #     preds=preds,
            #     target=labels,
            #     beta=config["f_beta"],
            #     tgt_vocab_size=vocab_size,
            #     pad_index=pad_token_id,
            #     device=device
            #     )
        else:
            recall = torcheval_recall(
                input=preds,
                target=labels,
                device=device
            )

            precision = torcheval_precision(
                input=preds,
                target=labels,
                device=device
            )

            # f_05 = torcheval_f_beta(
            #     recall=recall,
            #     precision=precision,
            #     beta=config["f_beta"]
            # )

        bleus = torchtext_bleu_score(refs=expected,
                                    cands=predicted)
        
        recall = recall.item()
        precision = precision.item()
        # f_05 = f_05.item()
        print(f"{recall = }")
        print(f"{precision = }")
        # print(f"{f_05 = }")
                
        return bleus, recall, precision