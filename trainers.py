from .utils.figures import (
    LossFigure,
)
from tqdm import tqdm
import torch
from .utils.folders import (
    get_weights_file_path,
)
import os
import json
from .val import validate

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
        loss_train_step_figure: LossFigure,
        loss_val_step_figure: LossFigure,
        loss_train_epoch_figure: LossFigure,
        loss_val_epoch_figure: LossFigure,
        rouge_1_epoch_figure: LossFigure,
        rouge_2_epoch_figure: LossFigure,
        rouge_l_epoch_figure: LossFigure,
        model_folder_name: str,
        model_base_name: str,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        state: dict=None,
    ):
        self.config = config
        self.model = model
        print(f"model: {model}")
        self.optimizer = optimizer
        self.device = device
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.loss_train_step_figure = loss_train_step_figure
        self.loss_val_step_figure = loss_val_step_figure
        self.loss_train_epoch_figure = loss_train_epoch_figure
        self.loss_val_epoch_figure = loss_val_epoch_figure
        self.rouge_1_epoch_figure = rouge_1_epoch_figure
        self.rouge_2_epoch_figure = rouge_2_epoch_figure
        self.rouge_l_epoch_figure = rouge_l_epoch_figure
        self.model_folder_name = model_folder_name
        self.model_base_name = model_base_name
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.global_step = 0
        self.global_epoch = 0
        self.max_epoch = config["max_epoch"]
        self.max_global_step = config["max_global_step"]
        if config["use_scheduler"]:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda step: self.lambda_lr(),
            )
        self.step_accumulation = config["step_accumulation"]
        if state is not None:
            self.model.load_state_dict(state["model_state_dict"])
            self.global_step = state["global_step"]
            self.global_epoch = state["global_epoch"]
            if self.config["continue_step"] == False:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                if self.config["use_scheduler"]:
                    self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])

    def lambda_lr(
        self,
    ):
        global_step = max(self.global_step, 1)
        return (self.config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * self.config["warmup_steps"] ** (-1.5))

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
                if self.config["use_scheduler"]:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.model.zero_grad(set_to_none=True)

        self.loss_train_epoch_figure.update(
            value=sum_loss / len(self.train_dataloader),
            step=self.global_epoch,
        )

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
        self.loss_val_epoch_figure.update(
            value=sum_loss / len(self.val_dataloader),
            step=self.global_epoch,
        )
        
    def test(
        self,
        epoch: int,
    ):
        beam_size = self.config["beams"][-1]
        ans = validate(
            model=self.model,
            config=self.config,
            beam_size=beam_size,
            val_dataloader=self.test_dataloader,
        )
        for i in range(len(ans)):
            res = ans[i]
            for name, value in res.items():
                if value is None:
                    continue
                if name == "rouge1_fmeasure":
                    self.rouge_1_epoch_figure.update(
                        value=value,
                        step=epoch,
                    )
                    print(f"rouge1_fmeasure: {value}")
                elif name == "rouge2_fmeasure":
                    self.rouge_2_epoch_figure.update(
                        value=value,
                        step=epoch,
                    )
                    print(f"rouge2_fmeasure: {value}")
                elif name == "rougeL_fmeasure":
                    self.rouge_l_epoch_figure.update(
                        value=value,
                        step=epoch,
                    )
                    print(f"rougeL_fmeasure: {value}")
                print(f"{name}: {value}")

    def save_checkpoint(
        self,
    ):
        model_filename = get_weights_file_path(
            model_folder_name=self.model_folder_name,
            model_base_name=self.model_base_name,    
            step=self.global_step
        )

        if self.config["use_scheduler"]:
            torch.save({
                "global_step": self.global_step,
                "global_epoch": self.global_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
            }, model_filename)
        else:
            torch.save({
                "global_step": self.global_step,
                "global_epoch": self.global_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
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
        self.rouge_1_epoch_figure.save()
        self.rouge_2_epoch_figure.save()
        self.rouge_l_epoch_figure.save()

    def train_loop(
        self,
    ):
        torch.cuda.empty_cache()
        initial_epoch = self.global_epoch + 1
        for epoch in range(initial_epoch, self.max_epoch + 1):
            if self.global_step + 1 > self.max_global_step:
                break
            self.global_epoch += 1
            self.train(epoch)
            self.val(epoch)
            self.test(epoch)
            self.save_figure()
            self.save_checkpoint()

class BartTrainerMultiGPU:
    def __init__(
        self,
        config: dict,
        model,
        optimizer,
        tokenizer_src,
        tokenizer_tgt,
        loss_train_step_figure: LossFigure,
        loss_val_step_figure: LossFigure,
        loss_train_epoch_figure: LossFigure,
        loss_val_epoch_figure: LossFigure,
        model_folder_name: str,
        model_base_name: str,
        train_dataloader,
        val_dataloader,
        state: dict=None,
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
        self.global_step = 0
        self.global_epoch = 0
        self.max_epoch = config["max_epoch"]
        self.max_global_step = config["max_global_step"]
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: self.lambda_lr(),
        )
        self.step_accumulation = config["step_accumulation"]
        if state is not None:
            self.model.load_state_dict(state["model_state_dict"])
            self.global_step = state["global_step"]
            self.global_epoch = state["global_epoch"]
            if self.config["continue_step"] == False:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])

    def lambda_lr(
        self,
    ):
        global_step = max(self.global_step, 1)
        return (self.config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * self.config["warmup_steps"] ** (-1.5))

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

            with self.model.no_sync():
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
        model = self.model.module
        model_filename = get_weights_file_path(
            model_folder_name=self.model_folder_name,
            model_base_name=self.model_base_name,    
            step=self.global_step,
        )

        torch.save({
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "model_state_dict": model.state_dict(),
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
        for epoch in range(self.global_epoch + 1, self.max_epoch + 1):
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