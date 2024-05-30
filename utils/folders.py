import os
from pathlib import Path

# create dirs
def create_dirs(dir_paths: list):
    created_dirs = []
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        created_dirs.append(dir_path)
    
    # message
    print("Created:")
    for name_dir in created_dirs:
        print(name_dir)
    print("====================================")

# file path
def get_weights_file_path(model_folder_name: str, model_base_name: str, step: int):
    model_name = f"{model_base_name}_{step:010d}.pt"
    return f"{model_folder_name}/{model_name}"

def weights_file_path(model_folder_name: str, model_base_name: str):
    model_filename = f"{model_base_name}*"
    weights_files = list(Path(model_folder_name).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files

__all__ = ["create_dirs", "get_weights_file_path", "weights_file_path"]