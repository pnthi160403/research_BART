import torch
import torch
from tqdm import tqdm

from .model import get_bart_model, save_model, save_config
from .prepare_dataset import get_dataloader, read_tokenizer
from .utils import set_seed, create_dirs, lambda_lr, get_weights_file_path, weights_file_path, draw_graph, draw_multi_graph
from transformers import  BartModel, BartConfig

def train(config):
    # create dirs
    create_dirs(
        config=config,
        dirs=["model_folder", "log_dir", "tokenizer_dir"]
    )
    
    # set seed
    set_seed()

    # device
    device = config["device"]

    # read tokenizer
    tokenizer = read_tokenizer(config)

    # BART model
    model = get_bart_model(
        config=config,
        tokenizer=tokenizer,
    ).to(device)

    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        config=config,
    )

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
        betas=config["betas"]
    )

    global_step = 0
    initial_epoch = 0

    preload = config["preload"]

    # get lr schduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda global_step: lambda_lr(
            global_step=global_step,
            config=config
        )
    )

    # load model
    model_filename = (str(weights_file_path(config)[-1]) if weights_file_path(config) else None) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        print(f"Loaded model from {model_filename}")
    else:
        print("No model to preload, start training from scratch")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id("<pad>"),
        label_smoothing=config["label_smoothing"],
    ).to(device)

    losses_train = []
    losses_val = []
    losses_train_step = []
    losses_val_step = []
    for epoch in range(initial_epoch, config['epochs']):
        torch.cuda.empty_cache()
        # train
        sum_loss_train = 0
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Trainning Epoch {epoch:02d}")
        for batch in batch_iterator:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_attention_mask = (src != tokenizer.token_to_id("<pad>")).type(torch.int64)
            tgt_attention_mask = (tgt != tokenizer.token_to_id("<pad>")).type(torch.int64)
            label = batch['label'].to(device)
            
            logits = model(
                input_ids=src,
                attention_mask=src_attention_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_attention_mask,
            )
            
            loss = loss_fn(logits.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            sum_loss_train += loss.item()
            losses_train_step.append(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            break

        # val
        with torch.no_grad():
            sum_loss_val = 0
            model.eval()
            batch_iterator = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}")
            for batch in batch_iterator:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                src_attention_mask = (src != tokenizer.token_to_id("<pad>")).type(torch.int64)
                tgt_attention_mask = (tgt != tokenizer.token_to_id("<pad>")).type(torch.int64)
                label = batch['label'].to(device)
                
                logits = model(
                    input_ids=src,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=tgt_attention_mask,
                )
                
                loss = loss_fn(logits.view(-1, tokenizer.get_vocab_size()), label.view(-1))
                sum_loss_val += loss.item()
                losses_val_step.append(loss.item())
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                break

        losses_train.append(sum_loss_train / len(train_dataloader))
        losses_val.append(sum_loss_val / len(val_dataloader))

    # save model
    if config["pretrain"]:
        save_model(
            model=model.bart_model,
            epoch=epoch,
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config
        )
    else:
        save_model(
            model=model,
            epoch=epoch,
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config
        )

    # save config
    save_config(
        config=config,
        epoch=epoch
    )

    # draw graph loss
    # train and val
    draw_multi_graph(
        config=config,
        xlabel="Value Loss",
        ylabel="Epoch",
        title="Loss",
        all_data=[
            (losses_train, "Train"),
            (losses_val, "Val")
        ]
    )
    # train step
    draw_graph(
        config=config,
        title="Loss train",
        xlabel="Loss",
        ylabel="Epoch",
        data=losses_train_step
    )

    # val step
    draw_graph(
        config=config,
        title="Loss val",
        xlabel="Loss",
        ylabel="Epoch",
        data=losses_val_step
    )

    # debug
    bart_config = BartConfig(
        d_model=config["d_model"],
        encoder_layes=config["encoder_layes"],
        decoder_layers=config["decoder_layers"],
        encoder_attention_heads=config["encoder_attention_heads"],
        decoder_attention_heads=config["decoder_attention_heads"],
        decoder_ffn_dim=config["decoder_ffn_dim"],
        encoder_ffn_dim=config["encoder_ffn_dim"],
        activation_function=config["activation_function"],
        dropout=config["dropout"],
        attention_dropout=config["attention_dropout"],
        activation_dropout=config["activation_dropout"],
        classifier_dropout=config["classifier_dropout"],
        max_position_embeddings=config["max_position_embeddings"],
        init_std=config["init_std"],
        encoder_layerdrop=config["encoder_layerdrop"],
        decoder_layerdrop=config["decoder_layerdrop"],
        scale_embedding=config["scale_embedding"],
        eos_token_id=tokenizer.token_to_id("</s>"),
        forced_bos_token_id=tokenizer.token_to_id("<s>"),
        forced_eos_token_id=tokenizer.token_to_id("</s>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
        num_beams=config["num_beams"],
        vocab_size=tokenizer.get_vocab_size()
    )

    model_filename = get_weights_file_path(config, f"{epoch:02d}")
    model = BartModel(bart_config)
    model.load_state_dict(torch.load(model_filename)["model_state_dict"])

    return model