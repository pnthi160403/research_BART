import torch
import torch
from tqdm import tqdm

from .model import save_model, save_config, GET_MODEL
from .prepare_dataset import get_dataloader, read_tokenizer
from .utils import set_seed, create_dirs, lambda_lr, get_weights_file_path, weights_file_path, draw_graph, draw_multi_graph, read, write

def train(config):
    # create dirs
    create_dirs(
        config=config,
        dirs=["model_folder", "log_dir", "log_files"]
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

    if global_step == 0:
        write(config["loss_train"], [])
        write(config["loss_val"], [])
        write(config["loss_train_step"], [])
        write(config["loss_val_step"], [])
        write(config["timestep_train"], [])
        write(config["timestep_val"], [])
        write(config["timestep_train_and_val"], [])

    losses_train = read(config["loss_train"])
    losses_val = read(config["loss_val"])
    losses_train_step = read(config["loss_train_step"])
    losses_val_step = read(config["loss_val_step"])

    timestep_train = read(config["timestep_train"]) # Ox for train
    timestep_val = read(config["timestep_val"]) # Ox for val
    timestep_train_and_val = read(config["timestep_train_and_val"]) # Ox for train and val

    while global_step < config["num_steps"]:
        torch.cuda.empty_cache()
        # train
        sum_loss_train = 0
        model.train()
        # shuffle dataloader
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(
            config=config,
        )

        batch_iterator = tqdm(train_dataloader, desc="Trainning")
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
            batch_iterator.set_postfix({
                "loss": f"{loss.item():6.3f}",
                "global_step": f"{global_step:010d}"
            })
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            timestep_train.append(global_step)

            if global_step >= config["num_steps"] or global_step % config["val_steps"] == 0:
                break

        # val
        with torch.no_grad():
            sum_loss_val = 0
            model.eval()
            batch_iterator = tqdm(val_dataloader, desc="Validating")

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
                timestep_val.append(global_step)

                batch_iterator.set_postfix({
                    "loss": f"{loss.item():6.3f}",
                    "global_step": f"{global_step:010d}"
                })
            
            if global_step % config["val_steps"] == 0:
                losses_train.append(sum_loss_train / len(train_dataloader))
            else:
                losses_train.append(sum_loss_train / (global_step % config["val_steps"]))
            losses_val.append(sum_loss_val / len(val_dataloader))

            timestep_train_and_val.append(global_step)

        if global_step >= config["num_steps"]:
            break

    # save model
    if config["pretrain"]:
        save_model(
            model=model.bart_model,
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            save_model="bart"
        )
    save_model(
        model=model,
        global_step=global_step,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config
    )

    # save config
    save_config(
        config=config,
        global_step=global_step,
    )

    # save log
    write(config["loss_train"], losses_train)
    write(config["loss_val"], losses_val)
    write(config["loss_train_step"], losses_train_step)
    write(config["loss_val_step"], losses_val_step)
    write(config["timestep_train"], timestep_train)
    write(config["timestep_val"], timestep_val)
    write(config["timestep_train_and_val"], timestep_train_and_val)

    # draw graph loss
    # train and val
    draw_multi_graph(
        config=config,
        xlabel="Step",
        ylabel="Loss value",
        title="Loss",
        all_data=[
            (losses_train, "Train"),
            (losses_val, "Val")
        ],
        steps=timestep_train_and_val
    )
    # train step
    draw_graph(
        config=config,
        title="Loss train",
        xlabel="Step",
        ylabel="Loss value",
        data=losses_train_step,
        steps=timestep_train
    )

    # val step
    draw_graph(
        config=config,
        title="Loss val",
        xlabel="Step",
        ylabel="Loss value",
        data=losses_val_step,
        steps=timestep_val
    )