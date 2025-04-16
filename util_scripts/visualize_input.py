import pickle
import argparse
from pathlib import Path
from diffusion_planner.utils.visualize_input import visualize_inputs
from diffusion_planner.utils.config import Config
from copy import deepcopy
from shutil import rmtree
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("args_json", type=Path)
    parser.add_argument("--num", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    args_json = args.args_json
    num = args.num

    ext = input_path.suffix
    config_obj = Config(args_json)

    if ext == ".pkl":
        with open(input_path, "rb") as f:
            data = pickle.load(f)
        bs = -1
        for key, value in data.items():
            print(f"{key}\t{value.shape}")
            bs = value.shape[0] if bs == -1 else bs
            assert bs == value.shape[0], "Batch size mismatch"
    elif ext == ".npz":
        loaded = np.load(input_path)
        # add batch size axis
        bs = 1
        data = {}
        for key, value in loaded.items():
            print(f"{key}\t{value.dtype}\t{value.shape}")
            if key == "map_name" or key == "token":
                continue
            data[key] = torch.tensor(np.expand_dims(value, axis=0))
        data = config_obj.observation_normalizer(data)

    save_dir = Path("./visualize")
    rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True)
    for b in range(min(bs, num)):
        inputs = deepcopy(data)
        for key, value in inputs.items():
            inputs[key] = value[b : b + 1]
        save_path = save_dir / f"input_{b:08d}.png"
        visualize_inputs(inputs, config_obj.observation_normalizer, save_path)
        print(f"Saved to {save_path}")
