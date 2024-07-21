import torch
from tqdm import tqdm
from .generate import generate
from .models.utils import (
    get_cosine_similarity,
)
from .utils.folders import (
    write_json,
    read_json,
)
from .utils.figures import (
    zip_directory,
)
from .utils.search import (
    DIVERSE_BEAM_SEARCH,
    BEAM_SEARCH,
    NEURAL_EMBEDDING_TYPE_DIVERSITY,
)
from torch.nn.utils.rnn import pad_sequence
from .utils.tokenizers import read_tokenizer
from .utils.metrics import (
    torchmetrics_accuracy,
    torchmetrics_recall,
    torchmetrics_precision,
    torchmetrics_f_beta,
    torchmetrics_rouge,
    torcheval_recall,
    torcheval_precision,
    torcheval_f_beta,
    torchtext_bleu_score
)

def validate(model, config, beam_size, val_dataloader, num_example: int=20):
    model.eval()
    device = config["device"]
    # get sub test id
    sub_test_id = config["sub_test_id"]
    
    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )

    # get cosine similarity
    decoder_embeds_matrix = torch.tensor(model.decoder_inputs_embeds.embed_tokens.weight.data.clone().detach().cpu().numpy()).to(device)
    if config["type_diversity_function"] == NEURAL_EMBEDDING_TYPE_DIVERSITY:
        top_cosine_similarity_indices = get_cosine_similarity(
            path=config["cosine_similarity_path"],
            vocab_size=config["tgt_vocab_size"],
            k=config["top_k_cosine_similarity"],
            decoder_embeds_matrix=decoder_embeds_matrix,
            eos_token_id=tokenizer_tgt.token_to_id("</s>")
        )
    else:
        top_cosine_similarity_indices = None
        
    vocab_size=tokenizer_tgt.get_vocab_size()
    pad_token_id = tokenizer_src.token_to_id("<pad>")

    with torch.no_grad():

        source_texts = []
        expected = []
        predicted = []

        count = 0

        labels = []
        preds = []

        rouge_preds = []
        rouge_targets = []
        
        batch_iterator = tqdm(val_dataloader, desc=f"Testing model...")
        for batch in batch_iterator:
            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]

            preds_ids = generate(
                model=model,
                config=config,
                beam_size=beam_size,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                src=src_text,
                top_cosine_similarity_indices=top_cosine_similarity_indices,
            )
            if config["type_search"] in [BEAM_SEARCH, DIVERSE_BEAM_SEARCH]:
                pred_ids = preds_ids[0].tgt.squeeze()
            
            pred_text = tokenizer_tgt.decode(
                pred_ids.detach().cpu().numpy(),
                skip_special_tokens=True,
            )

            rouge_preds.append(pred_text)
            rouge_targets.append(tgt_text)  
            
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

            count += 1

            print_step = max(len(val_dataloader) // num_example, 1)
            
            if count % print_step == 0:
                print()
                print(f"{f'SOURCE: ':>12}{src_text}")
                print(f"{f'TARGET: ':>12}{tgt_text}")
                for i in range(len(preds_ids)):
                    text = tokenizer_tgt.decode(
                        preds_ids[i].tgt.squeeze().detach().cpu().numpy(),
                        skip_special_tokens=True,
                    )
                    # print(f"{f'TOKENS TARGET {i}: ':>12}{[tokenizer_tgt.encode(text).tokens]}")
                    # print(f"{f'TOKENS IDS {i}: ':>12}{preds_ids[i].tgt.squeeze().detach().cpu().numpy()}")
                    print(f"{f'PREDICTED {i}: ':>12}{text}")
                    # print()
                if config["use_bleu"]:
                    scores = torchtext_bleu_score(refs=[[tgt_text.split()]],
                                            cands=[pred_text.split()])
                    print(f'BLEU OF SENTENCE {count}')
                    for i in range(0, len(scores)):
                        print(f'BLEU_{i + 1}: {scores[i]}')
                    
                if not config["use_pytorch_metric"]:
                    if config["use_recall"]:
                        recall = torchmetrics_recall(
                            preds=pred_ids,
                            target=label_ids,
                            tgt_vocab_size=vocab_size,
                            pad_index=pad_token_id,
                            device=device
                        )
                        recall = recall.item()
                        print(f"{recall = }")
                    if config["use_precision"]:
                        precision = torchmetrics_precision(
                            preds=pred_ids,
                            target=label_ids,
                            tgt_vocab_size=vocab_size,
                            pad_index=pad_token_id,
                            device=device
                        )
                        precision = precision.item()
                        print(f"{precision = }")
                else:
                    if config["use_recall"]:
                        recall = torcheval_recall(
                            input=pred_ids,
                            target=label_ids,
                            device=device
                        )
                        recall = recall.item()
                        print(f"{recall = }")
                    if config["use_precision"]:
                        precision = torcheval_precision(
                            input=pred_ids,
                            target=label_ids,
                            device=device
                        )
                        precision = precision.item()
                        print(f"{precision = }")

            
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        recall, precision, rouges = None, None, None

        if not config["use_pytorch_metric"]:
            if config["use_recall"]:
                recall = torchmetrics_recall(
                    preds=preds,
                    target=labels,
                    tgt_vocab_size=vocab_size,
                    pad_index=pad_token_id,
                    device=device
                )
            if config["use_precision"]:
                precision = torchmetrics_precision(
                    preds=preds,
                    target=labels,
                    tgt_vocab_size=vocab_size,
                    pad_index=pad_token_id,
                    device=device
                )
            if config["use_rouge"]:
                rouges = torchmetrics_rouge(
                    preds=rouge_preds,
                    target=rouge_targets,
                    device=device
                )
                write_json(
                    file_path=config["generated_dir"] + f"/rouge_preds_{sub_test_id}.json",
                    data=rouge_preds,
                )
                write_json(
                    file_path=config["generated_dir"] + f"/rouge_targets_{sub_test_id}.json",
                    data=rouge_targets,
                )
        else:
            if config["use_recall"]:
                recall = torcheval_recall(
                    input=preds,
                    target=labels,
                    device=device
                )
            if config["use_precision"]:
                precision = torcheval_precision(
                    input=preds,
                    target=labels,
                    device=device
                )

        if config["use_bleu"]:
            bleus = torchtext_bleu_score(refs=expected,
                                        cands=predicted)
            
        zip_directory(
            directory_path=config["generated_dir"],
            output_zip_path=f"{config['generated_dir']}_{config['sub_test_id']}.zip",
        )
        
        res = {}
        if config["use_bleu"]:
            for i in range(0, len(bleus)):
                res[f"bleu_{i+1}"] = bleus[i]
        if recall is not None and config["use_recall"]:
            res["recall"] = recall.item()
        if precision is not None and config["use_precision"]:
            res["precision"] = precision.item()
        if rouges is not None and config["use_rouge"]:
            for key, val in rouges.items():
                res[key] = val.item()
        return [res]