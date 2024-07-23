from .utils.figures import (
    LossFigure,
)
from tqdm import tqdm
import torch
from .utils.folders import (
    join_base,
    read,
    write,
    get_weights_file_path,
)
import os
import json

# get optimizer lambda lr
def lambda_lr(global_step: int, config):
    global_step = max(global_step, 1)
    return (config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * config["warmup_steps"] ** (-1.5))

class BartTrainerSingleGPU:
    def __init__(
        self,
        config: dict,
        model,
        optimizer,
        device,
        tokenizer_src,
        tokenizer_tgt,
        # lr_scheduler,
        loss_train_step_figure: LossFigure,
        loss_val_step_figure: LossFigure,
        loss_train_epoch_figure: LossFigure,
        loss_val_epoch_figure: LossFigure,
        model_folder_name: str,
        model_base_name: str,
        train_dataloader,
        val_dataloader,
        global_step: int,
        global_epoch: int,
        max_global_step: int,
        max_epoch: int,
        step_accumulation: int=1,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.loss_train_step_figure = loss_train_step_figure
        self.loss_val_step_figure = loss_val_step_figure
        self.loss_train_epoch_figure = loss_train_epoch_figure
        self.loss_val_epoch_figure = loss_val_epoch_figure
        self.model_folder_name = model_folder_name
        self.model_base_name = model_base_name
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.global_step = global_step
        self.global_epoch = global_epoch
        self.max_global_step = max_global_step
        self.max_epoch = max_epoch
        # self.lr_scheduler = lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: lambda_lr(
                global_step=self.global_epoch,
                config=config
            )
        )
        self.step_accumulation = step_accumulation

    def train(
        self,
        epoch: int,
    ):
        self.model.train()
        batch_iterator = tqdm(self.train_dataloader, desc=f"Train {epoch}")
        sum_loss = 0
        sum_loss_step_accumulation = 0
        for step, batch in enumerate(batch_iterator):
            if self.global_step + 1 > self.max_global_step:
                break
        
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_attention_mask = (src != self.tokenizer_src.token_to_id("<pad>")).type(torch.int64)
            tgt_attention_mask = (tgt != self.tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
            label = batch['label'].to(self.device)

            logits, loss = self.model(
                input_ids=src,
                attention_mask=src_attention_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_attention_mask,
                label=label,
            )
            loss.backward()
            sum_loss += loss.item()
            sum_loss_step_accumulation += loss.item()

            if (step + 1) % self.step_accumulation == 0:
                self.global_step += 1
                self.loss_train_step_figure.update(
                    value=sum_loss_step_accumulation / self.step_accumulation,
                    step=self.global_step,
                )
                batch_iterator.set_postfix({
                    "loss": f"{sum_loss_step_accumulation / self.step_accumulation:6.3f}",
                })
                sum_loss_step_accumulation = 0
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad(set_to_none=True)

        return sum_loss / len(self.train_dataloader)

    def val(
        self,
        epoch: int,
    ):
        self.model.eval()
        with torch.no_grad():
            batch_iterator = tqdm(self.val_dataloader, desc=f"Val {epoch}")
            sum_loss = 0
            for step, batch in enumerate(batch_iterator):
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                src_attention_mask = (src != self.tokenizer_src.token_to_id("<pad>")).type(torch.int64)
                tgt_attention_mask = (tgt != self.tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
                label = batch['label'].to(self.device)

                logits, loss = self.model(
                    input_ids=src,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=tgt_attention_mask,
                    label=label,
                )
                sum_loss += loss.item()
                self.loss_val_step_figure.update(
                    value=loss.item(),
                    step=len(self.loss_val_step_figure.loss_step) + 1,
                )
                batch_iterator.set_postfix({
                    "loss": f"{loss.item():6.3f}",
                })
            return sum_loss / len(self.val_dataloader)

    def save_checkpoint(
        self,
    ):
        model_filename = get_weights_file_path(
            model_folder_name=self.model_folder_name,
            model_base_name=self.model_base_name,    
            step=self.global_step
        )

        torch.save({
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
        }, model_filename)
        
        print(f"Saved model at {model_filename}")

        config_filename = f"{self.config['config_dir']}/config_{self.global_step:010d}.json"
        with open(config_filename, "w") as f:
            json.dump(self.config, f)
        print(f"Saved config at {config_filename}")

    def save_figure(
        self,
    ):
        self.loss_train_step_figure.save()
        self.loss_val_step_figure.save()
        self.loss_train_epoch_figure.save()
        self.loss_val_epoch_figure.save()

    def train_loop(
        self,
    ):
        torch.cuda.empty_cache()
        for epoch in range(self.global_epoch + 1, self.max_epoch):
            if self.global_epoch + 1 > self.max_epoch or self.global_step + 1 > self.max_global_step:
                break
            self.global_epoch += 1
            mean_loss_train_epoch = self.train(epoch)
            mean_loss_val_epoch = self.val(epoch)
            self.loss_train_epoch_figure.update(
                value=mean_loss_train_epoch,
                step=self.global_epoch,
            )
            self.loss_val_epoch_figure.update(
                value=mean_loss_val_epoch,
                step=self.global_epoch,
            )
            self.save_checkpoint()

class BartTrainerMultiGPU:
    def __init__(
        self,
        config: dict,
        model,
        optimizer,
        tokenizer_src,
        tokenizer_tgt,
        lr_scheduler,
        loss_train_step_figure: LossFigure,
        loss_val_step_figure: LossFigure,
        loss_train_epoch_figure: LossFigure,
        loss_val_epoch_figure: LossFigure,
        model_folder_name: str,
        model_base_name: str,
        train_dataloader,
        val_dataloader,
        global_step: int,
        global_epoch: int,
        max_global_step: int,
        max_epoch: int,
        step_accumulation: int=1,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.loss_train_step_figure = loss_train_step_figure
        self.loss_val_step_figure = loss_val_step_figure
        self.loss_train_epoch_figure = loss_train_epoch_figure
        self.loss_val_epoch_figure = loss_val_epoch_figure
        self.model_folder_name = model_folder_name
        self.model_base_name = model_base_name
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.global_step = global_step
        self.global_epoch = global_epoch
        self.max_global_step = max_global_step
        self.max_epoch = max_epoch
        self.lr_scheduler = lr_scheduler
        self.step_accumulation = step_accumulation

    def train(
        self,
        epoch: int,
    ):
        torch.cuda.empty_cache()
        self.model.train()
        batch_iterator = tqdm(self.train_dataloader, desc=f"Train {epoch}")
        sum_loss = 0
        sum_loss_step_accumulation = 0
        for step, batch in enumerate(batch_iterator):
            if self.global_step + 1 > self.max_global_step:
                break
        
            src = batch["src"].to(int(os.environ["LOCAL_RANK"]))
            tgt = batch["tgt"].to(int(os.environ["LOCAL_RANK"]))
            src_attention_mask = (src != self.tokenizer_src.token_to_id("<pad>")).type(torch.int64)
            tgt_attention_mask = (tgt != self.tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
            label = batch['label'].to(int(os.environ["LOCAL_RANK"]))

            logits, loss = self.model(
                input_ids=src,
                attention_mask=src_attention_mask,
                decoder_input_ids=tgt,
                decoder_attention_mask=tgt_attention_mask,
                label=label,
            )
            loss.backward()
            sum_loss += loss.item()
            sum_loss_step_accumulation += loss.item()

            if (step + 1) % self.step_accumulation == 0:
                self.global_step += 1
                self.loss_train_step_figure.update(
                    value=sum_loss_step_accumulation / self.step_accumulation,
                    step=self.global_step,
                )
                batch_iterator.set_postfix({
                    "loss": f"{sum_loss_step_accumulation / self.step_accumulation:6.3f}",
                })
                sum_loss_step_accumulation = 0
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad(set_to_none=True)

        return sum_loss / len(self.train_dataloader)

    def val(
        self,
        epoch: int,
    ):
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.model.eval()
            batch_iterator = tqdm(self.val_dataloader, desc=f"Val {epoch}")
            sum_loss = 0
            for step, batch in enumerate(batch_iterator):
                src = batch["src"].to(int(os.environ["LOCAL_RANK"]))
                tgt = batch["tgt"].to(int(os.environ["LOCAL_RANK"]))
                src_attention_mask = (src != self.tokenizer_src.token_to_id("<pad>")).type(torch.int64)
                tgt_attention_mask = (tgt != self.tokenizer_tgt.token_to_id("<pad>")).type(torch.int64)
                label = batch['label'].to(int(os.environ["LOCAL_RANK"]))

                logits, loss = self.model(
                    input_ids=src,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=tgt,
                    decoder_attention_mask=tgt_attention_mask,
                    label=label,
                )
                sum_loss += loss.item()
                self.loss_val_step_figure.update(
                    value=loss.item(),
                    step=len(self.loss_val_step_figure.loss_step) + 1,
                )
                batch_iterator.set_postfix({
                    "loss": f"{loss.item():6.3f}",
                })
            return sum_loss / len(self.val_dataloader)

    def save_checkpoint(
        self,
    ):
        model_filename = get_weights_file_path(
            model_folder_name=self.model_folder_name,
            model_base_name=self.model_base_name,    
            step=self.global_step
        )

        torch.save({
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
        }, model_filename)
        
        print(f"Saved model at {model_filename}")

        config_filename = f"{self.config['config_dir']}/config_{self.global_step:010d}.json"
        with open(config_filename, "w") as f:
            json.dump(self.config, f)
        print(f"Saved config at {config_filename}")

    def save_figure(
        self,
    ):
        self.loss_train_step_figure.save()
        self.loss_val_step_figure.save()
        self.loss_train_epoch_figure.save()
        self.loss_val_epoch_figure.save()

    def train_loop(
        self,
    ):
        for epoch in range(self.global_epoch + 1, self.max_epoch):
            if self.global_epoch + 1 > self.max_epoch or self.global_step + 1 > self.max_global_step:
                break
            self.global_epoch += 1
            mean_loss_train_epoch = self.train(epoch)
            mean_loss_val_epoch = self.val(epoch)
            self.loss_train_epoch_figure.update(
                value=mean_loss_train_epoch,
                step=self.global_epoch,
            )
            self.loss_val_epoch_figure.update(
                value=mean_loss_val_epoch,
                step=self.global_epoch,
            )
            if int(os.environ["LOCAL_RANK"]) == 0:
                self.save_checkpoint()