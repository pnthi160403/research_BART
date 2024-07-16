import torch
import torch
from tqdm import tqdm

from .prepare_dataset.seq2seq import get_dataloader
from .utils.tokenizers import read_tokenizer
from .utils.figures import (
    draw_graph,
    draw_multi_graph,
    read,
    write,
    save_model,
    save_config,
    zip_directory,
)
from .utils.folders import (
    create_dirs,
    get_weights_file_path,
    weights_file_path,
)
from .utils.seed import set_seed
from .utils.tokenizers import read_tokenizer
from .utils.optimizers import (
    GET_OPTIMIZER,
)

from .models.get_instance_bart import get_model

# get optimizer lambda lr
def lambda_lr(global_step: int, config):
    global_step = max(global_step, 1)
    return (config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * config["warmup_steps"] ** (-1.5))

def train(config):
    # create dirs
    create_dirs(dir_paths=[config["log_dir"], config["model_folder_name"], config["log_files"], config["config_dir"]])
    
    # set seed
    set_seed(seed=config["seed"])

    # device
    device = config["device"]

    # big batch
    big_batch = config["big_batch"]
    step_accumulation = big_batch // config["batch_train"]

    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )
    config["src_vocab_size"] = tokenizer_src.get_vocab_size()
    config["tgt_vocab_size"] = tokenizer_tgt.get_vocab_size()
    config["pad_idx"] = tokenizer_src.token_to_id("<pad>")

    # BART model
    model = get_model(
        config=config,
        model_train=config["model_train"],
    ).to(device)

    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        batch_train=config["batch_train"],
        batch_val=config["batch_val"],
        batch_test=config["batch_test"],
        lang_src=config["lang_src"],
        lang_tgt=config["lang_tgt"],
        train_ds_path=config["train_ds_path"],
        val_ds_path=config["val_ds_path"],
        test_ds_path=config["test_ds_path"],
        max_num_val=config["max_num_val"],
        max_num_test=config["max_num_test"],
    )

    # optimizer
    optimizer_name = config["optimizer_name"]
    optimizer = GET_OPTIMIZER[optimizer_name](
        model=model,
        lr=config["lr"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
        betas=config["betas"]
    )

    global_step = 0
    global_val_step = 0

    preload = config["preload"]

    # get lr schduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lambda_lr(
            global_step=global_step,
            config=config
        )
    )

    # load model
    model_folder_name=config["model_folder_name"]
    model_base_name=config["model_base_name"]
    weights_files = weights_file_path(
        model_folder_name=model_folder_name,
        model_base_name=model_base_name,
    )
    if weights_files is not None:
        model_filename = (str(weights_files[-1]) if preload == 'latest' else get_weights_file_path(
            model_folder_name=model_folder_name,
            model_base_name=model_base_name,
        )) if preload else None
    else:
        model_filename = None
    if model_filename:
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        global_step = state["global_step"]
        global_val_step = state["global_val_step"]
        if config["continue_step"] == False:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
        print(f"Loaded model from {model_filename}")
    else:
        print("No model to preload, start training from scratch")

    if global_step == 0:
        write(config["loss_train"], []) # Oy for loss train in per epoch
        write(config["loss_val"], []) # Oy for loss val in per epoch
        write(config["loss_train_step"], []) # Oy for loss train in per step
        write(config["loss_val_step"], []) # Oy for loss val in per step
        write(config["learning_rate_step"], []) # Oy for learning rate in per step
        write(config["timestep_train"], []) # Ox for train
        write(config["timestep_val"], []) # Ox for val
        write(config["timestep_train_and_val"], []) # Ox for train and val
        write(config["timestep_lr"], []) # Ox for lr

    losses_train = read(config["loss_train"])
    losses_val = read(config["loss_val"])
    losses_train_step = read(config["loss_train_step"])
    losses_val_step = read(config["loss_val_step"])
    learning_rate_step = read(config["learning_rate_step"])

    timestep_train = read(config["timestep_train"])
    timestep_val = read(config["timestep_val"])
    timestep_train_and_val = read(config["timestep_train_and_val"])
    timestep_lr = read(config["timestep_lr"])

    i = 0
    while global_step < config["num_steps"]:
        torch.cuda.empty_cache()
        # train
        sum_loss_train = 0
        cnt_update_loss_train = 0
        model.train()
        # shuffle dataloader
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            batch_train=config["batch_train"],
            batch_val=config["batch_val"],
            batch_test=config["batch_test"],
            lang_src=config["lang_src"],
            lang_tgt=config["lang_tgt"],
            train_ds_path=config["train_ds_path"],
            val_ds_path=config["val_ds_path"],
            test_ds_path=config["test_ds_path"],
        )

        batch_iterator = tqdm(train_dataloader, desc="Trainning")
        for batch in batch_iterator:
            if global_step >= config["num_steps"]:
                break
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_attention_mask = (src != tokenizer_src.token_to_id("<pad>")).type(torch.int64)
            tgt_attention_mask = (tgt != tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
            label = batch['label'].to(device)

            logits, loss = model(
                input_ids=src,
                attention_mask=src_attention_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_attention_mask,
                label=label,
            )
            loss.backward()
            sum_loss_train += loss.item()
            cnt_update_loss_train += 1
            i += 1

            if i % step_accumulation == 0:
                global_step += 1
                current_lr = optimizer.param_groups[0]['lr']
                learning_rate_step.append(current_lr)
                timestep_lr.append(global_step)

                # loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                losses_train_step.append(loss.item())
                timestep_train.append(global_step)

                batch_iterator.set_postfix({
                    "loss": f"{loss.item():6.3f}",
                    "global_step": f"{global_step:010d}"
                })

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)

                if global_step % config["val_steps"] == 0:
                    # val
                    with torch.no_grad():
                        sum_loss_val = 0
                        cnt_update_loss_val = 0
                        model.eval()
                        # batch_iterator = tqdm(val_dataloader, desc="Validating")

                        # for batch in batch_iterator:
                        for batch in val_dataloader:
                            global_val_step += 1
                            src = batch["src"].to(device)
                            tgt = batch["tgt"].to(device)
                            src_attention_mask = (src != tokenizer_src.token_to_id("<pad>")).type(torch.int64)
                            tgt_attention_mask = (tgt != tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
                            label = batch['label'].to(device)
                            
                            logits, loss = model(
                                input_ids=src,
                                attention_mask=src_attention_mask,
                                decoder_input_ids=tgt,
                                decoder_attention_mask=tgt_attention_mask,
                                label=label,
                            )
                            
                            # loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                            sum_loss_val += loss.item()
                            cnt_update_loss_val += 1
                            losses_val_step.append(loss.item())
                            timestep_val.append(global_val_step)
                            
                    losses_train.append(sum_loss_train / cnt_update_loss_train)
                    losses_val.append(sum_loss_val / cnt_update_loss_val)
                    sum_loss_train = 0
                    sum_loss_val = 0
                    cnt_update_loss_train = 0
                    cnt_update_loss_val = 0

                    timestep_train_and_val.append(global_step)

    # save model
    save_model(
        model=model,
        global_step=global_step,
        global_val_step=global_val_step,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        model_folder_name=config["model_folder_name"],
        model_base_name=config["model_base_name"],
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
    write(config["learning_rate_step"], learning_rate_step)

    write(config["timestep_train"], timestep_train)
    write(config["timestep_val"], timestep_val)
    write(config["timestep_train_and_val"], timestep_train_and_val)
    write(config["timestep_lr"], timestep_lr)

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

    # learning rate step
    draw_graph(
        config=config,
        title="Learning rate",
        xlabel="Step",
        ylabel="Learning rate value",
        data=learning_rate_step,
        steps=timestep_lr
    )

    # zip directory
    zip_directory(
        directory_path=config["log_dir"],
        output_zip_path=config["log_dir_zip"],
    )
    zip_directory(
        directory_path=config["config_dir"],
        output_zip_path=config["config_dir_zip"],
    )
    zip_directory(
        directory_path=config["model_folder_name"],
        output_zip_path=config["model_folder_name_zip"],
    )