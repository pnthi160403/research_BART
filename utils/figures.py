import matplotlib.pyplot as plt
import pandas as pd
import torch
import json
import zipfile
import os
from .folders import (
    join_base,
    read,
    write,
    get_weights_file_path,
)

# figures
def draw_graph(config, title, xlabel, ylabel, data, steps):
    try:
        save_path = join_base(config['log_dir'], f"/{title}.png")
        plt.plot(steps, data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)

def draw_multi_graph(config, title, xlabel, ylabel, all_data, steps):
    try:
        save_path = join_base(config['log_dir'], f"/{title}.png")
        for data, info in all_data:
            plt.plot(steps, data, label=info)
            # add multiple legends
            plt.legend()

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
        plt.close()
    except Exception as e:
        print(e)

def figure_list_to_csv(config, column_names, data, name_csv):
    try:
        obj = {}
        for i in range(len(column_names)):
            if data[i] is not None:
                obj[str(column_names[i])] = data[i]

        data_frame = pd.DataFrame(obj, index=[0])
        save_path = join_base(config['log_dir'], f"/{name_csv}.csv")
        data_frame.to_csv(save_path, index=False)
        return data_frame
    except Exception as e:
        print(e)

def zip_directory(directory_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add file to zip, preserving the directory structure
                arcname = os.path.relpath(file_path, start=directory_path)
                zipf.write(file_path, arcname)

# save model
def save_model(model, global_step, global_val_step, optimizer, lr_scheduler, model_folder_name, model_base_name):
    model_filename = get_weights_file_path(
        model_folder_name=model_folder_name,
        model_base_name=model_base_name,    
        step=global_step
    )

    torch.save({
        "global_step": global_step,
        "global_val_step": global_val_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict()
    }, model_filename)
    
    print(f"Saved model at {model_filename}")

# save config
def save_config(config: dict, global_step: int):
    config_filename = f"{config['model_folder_name']}/config_{global_step:010d}.json"
    with open(config_filename, "w") as f:
        json.dump(config, f)
    print(f"Saved config at {config_filename}")

__all__ = [
    "read",
    "write",
    "draw_graph",
    "draw_multi_graph",
    "figure_list_to_csv",
    "save_model",
    "save_config",
    "zip_directory",
]