import os
import matplotlib.pyplot as plt
import pandas as pd
from .folders import (
    join_base,
    read,
    write
)

# figures
def draw_graph(config, title, xlabel, ylabel, data, steps):
    try:
        save_path = join_base(config['log_dir'], f"/{title}.png")
        plt.plot(steps, data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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

__all__ = ["read", "write", "draw_graph", "draw_multi_graph", "figure_list_to_csv"]