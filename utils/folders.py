import os
from pathlib import Path

def read(file_path):
    if not os.path.exists(file_path):
        return []
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            val = float(line.strip())
            data.append(val)
    return data

def write(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for value in data:
                file.write(f"{value}\n")
    except Exception as e:
        print(e)

def join_base(base_dir: str, path: str):
    return f"{base_dir}{path}"

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
    model_name = f"{model_base_name}{step:010d}.pt"
    return f"{model_folder_name}/{model_name}"

def weights_file_path(model_folder_name: str, model_base_name: str):
    model_filename = f"{model_base_name}*"
    weights_files = list(Path(model_folder_name).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files

__all__ = [
    "read",
    "write",
    "join_base",
    "create_dirs",
    "get_weights_file_path",
    "weights_file_path"
]