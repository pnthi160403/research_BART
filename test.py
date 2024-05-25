import torch
from .utils import set_seed, figure_list_to_csv, weights_file_path
from .prepare_dataset import get_dataloader, read_tokenizer
from .model import GET_MODEL
from .val import validate

def test(config):
    set_seed()
    device = config['device']
    device = torch.device(device)
    beams = config["beams"]

    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(config)

    # get dataloader
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config=config)

    # get model
    model_train = config["model_train"]
    get_model = GET_MODEL[model_train]
    model = get_model(
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt
    ).to(device)
        
    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]

    if model_filename:
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("No model to preload!")

    for beam_size in beams:
        bleus, recall, precision, f_05 = validate(
            model=model,
            config=config,
            beam_size=beam_size,
            val_dataloader=test_dataloader
        )

        data_frame = figure_list_to_csv(
            config=config,
            column_names=["bleu_1", "bleu_2", "bleu_3", "bleu_4", "recall", "precision", "f_05"],
            data=bleus + [recall, precision, f_05],
            name_csv=f"results_beam_{beam_size}"
        )

        print(data_frame)