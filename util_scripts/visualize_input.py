import argparse
from pathlib import Path
from diffusion_planner.utils.visualize_input import visualize_inputs
from diffusion_planner.utils.config import Config
from copy import deepcopy
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("args_json", type=Path)
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument("--num", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    args_json = args.args_json
    save_path = args.save_path
    num = args.num

    ext = input_path.suffix
    config_obj = Config(args_json)

    loaded = np.load(input_path)
    data = {}
    for key, value in loaded.items():
        print(f"{key}\t{value.dtype}\t{value.shape}")
        if key == "map_name" or key == "token":
            continue
        # add batch size axis
        data[key] = torch.tensor(np.expand_dims(value, axis=0))
    data = config_obj.observation_normalizer(data)

    inputs = deepcopy(data)
    for key, value in inputs.items():
        inputs[key] = value[0:1]
    visualize_inputs(inputs, config_obj.observation_normalizer, save_path)
    print(f"Saved to {save_path}")
