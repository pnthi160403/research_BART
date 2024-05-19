import torch
import torch
from tqdm import tqdm

from .model import save_model, save_config, GET_MODEL
from .prepare_dataset import get_dataloader, read_tokenizer
from .utils import set_seed, create_dirs, lambda_lr, get_weights_file_path, weights_file_path, draw_graph, draw_multi_graph

def train(config):
    # create dirs
    create_dirs(
        config=config,
        dirs=["model_folder", "log_dir"]
    )
    
    # set seed
    set_seed()

    # device
    device = config["device"]

    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(config=config)

    # BART model
    model_train = config["model_train"]
    get_model = GET_MODEL[model_train]
    model = get_model(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt
    ).to(device)

    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        config=config,
    )

    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        if config["continue_step"] == False:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        print(f"Loaded model from {model_filename}")
    else:
        print("No model to preload, start training from scratch")

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("<pad>"),
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
            src_attention_mask = (src != tokenizer_src.token_to_id("<pad>")).type(torch.int64)
            tgt_attention_mask = (tgt != tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
            label = batch['label'].to(device)
            
            logits = model(
                input_ids=src,
                attention_mask=src_attention_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_attention_mask,
            )
            
            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            sum_loss_train += loss.item()
            losses_train_step.append(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # val
        with torch.no_grad():
            sum_loss_val = 0
            model.eval()
            batch_iterator = tqdm(val_dataloader, desc=f"Validating Epoch {epoch:02d}")
            for batch in batch_iterator:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                src_attention_mask = (src != tokenizer_src.token_to_id("<pad>")).type(torch.int64)
                tgt_attention_mask = (tgt != tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
                label = batch['label'].to(device)
                
                logits = model(
                    input_ids=src,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=tgt_attention_mask,
                )
                
                loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                sum_loss_val += loss.item()
                losses_val_step.append(loss.item())
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        losses_train.append(sum_loss_train / len(train_dataloader))
        losses_val.append(sum_loss_val / len(val_dataloader))

    # save model
    if config["pretrain"]:
        save_model(
            model=model.bart_model,
            epoch=config['epochs'] - 1,
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            save_model="bart"
        )
    save_model(
        model=model,
        epoch=config['epochs'] - 1,
        global_step=global_step,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config
    )

    # save config
    save_config(
        config=config,
        epoch=config["epochs"] - 1,
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