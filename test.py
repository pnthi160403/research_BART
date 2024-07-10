import torch
from .val import validate
from .utils.seed import set_seed
from .utils.tokenizers import read_tokenizer
from .utils.folders import weights_file_path
from .utils.figures import figure_list_to_csv, zip_directory
from .prepare_dataset.seq2seq import get_dataloader
from .models.get_instance_bart import get_model

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
    config["src_vocab_size"] = tokenizer_src.get_vocab_size()
    config["tgt_vocab_size"] = tokenizer_tgt.get_vocab_size()
    config["pad_idx"] = tokenizer_src.token_to_id("<pad>")

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
        max_num_val=config["max_num_val"],
        max_num_test=config["max_num_test"],
    )

    # get model
    model = get_model(
        config=config,
        model_train=config["model_train"],
    ).to(device)
        
    model_filenames = weights_file_path(
        model_folder_name=config["model_folder_name"],
        model_base_name=config["model_base_name"],
    )
    model_filename = model_filenames[-1]

    if model_filename:
        print(f"Preloading model from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
    else:
        print("No model to preload!")

    for beam_size in beams:
        ans = validate(
            model=model,
            config=config,
            beam_size=beam_size,
            val_dataloader=test_dataloader
        )
        for i in range(len(ans)):
            res = ans[i]
            column_names = []
            data = []
            for name, value in res.items():
                if value is None:
                    continue
                column_names.append(name)
                data.append(value)

            data_frame = figure_list_to_csv(
                config=config,
                column_names=column_names,
                data=data,
                name_csv=f"results_beam_{beam_size}_prediction_{i}.csv"
            )

            zip_directory(
                directory_path=config["log_dir"],
                output_zip_path=config["log_dir_zip"]
            )

            print(f"Result test model in prediction {i} with beam size {beam_size}")
            print(data_frame)

# import torch
# from .val import validate
# from .utils.seed import set_seed
# from .utils.tokenizers import read_tokenizer
# from .utils.folders import weights_file_path
# from .utils.figures import figure_list_to_csv, zip_directory
# from .prepare_dataset.seq2seq import get_dataloader
# from .models.get_instance_bart import get_model

# def test(config):
#     # set seed
#     set_seed(seed=config['seed'])

#     device = config['device']
#     device = torch.device(device)
#     beams = config["beams"]

#     # read tokenizer
#     tokenizer_src, tokenizer_tgt = read_tokenizer(
#         tokenizer_src_path=config["tokenizer_src_path"],
#         tokenizer_tgt_path=config["tokenizer_tgt_path"],
#     )
#     config["src_vocab_size"] = tokenizer_src.get_vocab_size()
#     config["tgt_vocab_size"] = tokenizer_tgt.get_vocab_size()
#     config["pad_idx"] = tokenizer_src.token_to_id("<pad>")

#     # get dataloader
#     train_dataloader, val_dataloader, test_dataloader = get_dataloader(
#         tokenizer_src=tokenizer_src,
#         tokenizer_tgt=tokenizer_tgt,
#         batch_train=config["batch_train"],
#         batch_val=config["batch_val"],
#         batch_test=config["batch_test"],
#         lang_src=config["lang_src"],
#         lang_tgt=config["lang_tgt"],
#         train_ds_path=config["train_ds_path"],
#         val_ds_path=config["val_ds_path"],
#         test_ds_path=config["test_ds_path"],
#         max_num_val=config["max_num_val"],
#         max_num_test=config["max_num_test"],
#     )

#     # get model
#     model = get_model(
#         config=config,
#         model_train=config["model_train"],
#     ).to(device)
        
#     model_filenames = weights_file_path(
#         model_folder_name=config["model_folder_name"],
#         model_base_name=config["model_base_name"],
#     )
#     model_filename = model_filenames[-1]

#     if model_filename:
#         print(f"Preloading model from {model_filename}")
#         state = torch.load(model_filename)
#         model.load_state_dict(state['model_state_dict'])
#     else:
#         print("No model to preload!")

#     for beam_size in beams:
#         res = validate(
#             model=model,
#             config=config,
#             beam_size=beam_size,
#             val_dataloader=test_dataloader
#         )

#         column_names = []
#         data = []
#         for name, value in res.items():
#             if value is None:
#                 continue
#             column_names.append(name)
#             data.append(value)

#         data_frame = figure_list_to_csv(
#             config=config,
#             column_names=column_names,
#             data=data,
#             name_csv=f"results_beam_{beam_size}"
#         )

#         zip_directory(
#             directory_path=config["log_dir"],
#             output_zip_path=config["log_dir_zip"]
#         )

#         print(data_frame)