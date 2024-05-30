import torch
from .model import GET_MODEL
from .val import validate
from .utils.seed import set_seed
from .utils.tokenizers import read_tokenizer
from .utils.folders import weights_file_path
from .utils.figures import figure_list_to_csv
from .prepare_dataset.seq2seq import get_dataloader

def test(config):
    # set seed
    set_seed(seed=config['seed'])

    device = config['device']
    device = torch.device(device)
    beams = config["beams"]

    # read tokenizer
    tokenizer_src, tokenizer_tgt = read_tokenizer(
        tokenizer_src_path=config["tokenizer_src_path"],
        tokenizer_tgt_path=config["tokenizer_tgt_path"],
    )

    # get dataloader
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
        bleus, recall, precision = validate(
            model=model,
            config=config,
            beam_size=beam_size,
            val_dataloader=test_dataloader
        )

        data_frame = figure_list_to_csv(
            config=config,
            column_names=["bleu_1", "bleu_2", "bleu_3", "bleu_4", "recall", "precision"],
            data=bleus + [recall, precision],
            name_csv=f"results_beam_{beam_size}"
        )

        print(data_frame)