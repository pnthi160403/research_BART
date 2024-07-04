import torch
from .generate import generate
from .utils.tokenizers import read_tokenizer
from .utils.folders import weights_file_path
from .models.get_instance_bart import get_model
from .models.utils import load_model

def prepare_inference(config):
    device = config["device"]

    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )
    
    # get model
    model = get_model(
        config=config,
        model_train=config["model_train"],
    ).to(device)

    # get model_filename
    model_filename = weights_file_path(
        model_folder_name=config["model_folder_name"],
        model_base_name=config["model_base_name"],
    )[-1]

    model = load_model(
        model=model,
        checkpoint=model_filename,
    )

    return config, model, tokenizer_src, tokenizer_tgt

def inference(src, beam_size, prepare_inference):
    config, model, tokenizer_src, tokenizer_tgt = prepare_inference

    with torch.no_grad():
        model.eval()
    
        model_out = generate(
            model=model,
            config=config,
            beam_size=beam_size,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            src=src,
        )[0].tgt.squeeze()

        pred = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        return pred
    
def pipeline(config, src, beam_size):
    print("SOURCE:")
    print(src)
    print("PREDICT:")
    print(inference(
        src=src,
        beam_size=beam_size,
        prepare_inference=prepare_inference(config),
    ))

__all__ = [
    "prepare_inference",
    "inference",
    "pipeline"
]