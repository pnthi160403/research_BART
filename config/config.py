import torch
from ..utils import join_base

# TPU
# import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.serialization as xser
# import torch_xla.debug.metrics as met
# import torch_xla.debug.profiler as xp

# def check_tpu():
#     try:
#         device = xm.xla_device()
#         print(f"TPU is available. Device: {device}")
#         return True
#     except RuntimeError as e:
#         print("TPU is not available.")
#         return False

def get_config(base_dir: str=None):
    config = {}

    if not base_dir:
        config["base_dir"] = "./"
    else:
        config["base_dir"] = base_dir

    # Tokenizer
    config['tokenizer_tgt'] = None
    config['tokenizer_src'] = None
    config["use_tokenizer"] = "huggingface"
    config["special_tokens"] = [
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<mask>"
    ]
    config["vocab_size"] = 30000
    config['min_frequency'] = 2

    # Directories
    config["model_folder"] = join_base(config["base_dir"], "/model")
    config["model_basename"] = "model_"
    config["model_bart_basename"] = "bart_model_"
    config["model_inputs_embeds"] = "inputs_embeds_"
    config["model_decoder_inputs_embeds"] = "decoder_inputs_embeds_"
    config["model_out"] = "out_"
    
    config["preload"] = "latest"
    config["data"] = join_base(config["base_dir"], "/data")
    config["log_dir"] = join_base(config["base_dir"], "/log")
    config["log_files"] = join_base(config["log_dir"], "/log_files")
    config["loss_train"] = join_base(config["log_files"], "/loss_train.json")
    config["loss_val"] = join_base(config["log_files"], "/loss_val.json")
    config["loss_train_step"] = join_base(config["log_files"], "/loss_train_step.json")
    config["loss_val_step"] = join_base(config["log_files"], "/loss_val_step.json")
    config["timestep_train"] = join_base(config["log_files"], "/timestep_train.json")
    config["timestep_val"] = join_base(config["log_files"], "/timestep_val.json")
    config["timestep_train_and_val"] = join_base(config["log_files"], "/timestep_train_and_val.json")
    config["timestep_lr"] = join_base(config["log_files"], "/timestep_lr.json")

    # Dataset
    config["lang_src"] = "noise_vi"
    config["lang_tgt"] = "vi"
    config["train_ds"] = None
    config["val_ds"] = None
    config["test_ds"] = None
    config["corpus"] = None
    config["max_len"] = 100

    # Train
    config["model_train"] = "bart"
    config["step_train"] = None
    config["preload"] = "latest"
    config["pretrain"] = False
    config["continue_step"] = False
    
    # Trainning loop
    config["batch_train"] = 32
    config["batch_val"] = 32
    config["batch_test"] = 1
    config["num_steps"] = 15000
    config["val_steps"] = config["num_steps"] // 5

    # BART config model
    config["d_model"] = 768
    config["encoder_layers"] = 6
    config["decoder_layers"] = 6
    config["encoder_attention_heads"] = 12
    config["decoder_attention_heads"] = 12
    config["decoder_ffn_dim"] = 3072
    config["encoder_ffn_dim"] = 3072
    config["activation_function"] = "gelu"
    config["dropout"] = 0.1
    config["attention_dropout"] = 0.1
    config["activation_dropout"] = 0.1
    config["classifier_dropout"] = 0.0
    config["max_position_embeddings"] = config["max_len"] # The maximum sequence length
    config["init_std"] = 0.02 # Std for initializing all weight matrices
    config["encoder_layerdrop"] = 0.0 # Dropout encoder layer
    config["decoder_layerdrop"] = 0.0 # Dropout decoder layer
    config["scale_embedding"] = False # Scale embeddings with sqrt(d_model)
    config["num_beams"] = 4
    config["init_type"] = "normal"
    config["share_tgt_emb_and_out"] = False
    config["checkpoint"] = None

    # Optimizer Adam
    config["weight_decay"] = 0.0
    config["lr"] = 0.5
    config["eps"] = 1e-9
    config["betas"] = (0.9, 0.98)
    config["label_smoothing"] = 0.01

    # Scheduler (Noam decay)
    config["warmup_steps"] = 4000

    # Device
    # if check_tpu():
    #     config["device"] = xm.xla_device()
    # else:
    #     config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Metric
    config["f_beta"] = 0.5
    config["beams"] = [2]
    config["use_pytorch_metric"] = False

    return config

__all__ = [
    "get_config"
]