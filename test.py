import torch
from .utils import set_seed, figure_list_to_csv, weights_file_path
from .prepare_dataset import get_dataloader, read_tokenizer
from .model import get_bart_model
from .val import validate

def test(config):
    set_seed()
    device = "cpu"
    device = torch.device(device)
    beams = config["beams"]

    # read tokenizer
    tokenizer = read_tokenizer(config=config)

    # get dataloader
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config=config)

    # get model
    model = get_bart_model(
        config=config,
        tokenizer=tokenizer
    ).to(device)
        
    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]

    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

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
            name_csv=f"results_beam_{beam_size}.csv"
        )

        print(data_frame)