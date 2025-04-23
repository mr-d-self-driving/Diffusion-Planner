import argparse
from pathlib import Path
from diffusion_planner.utils.visualize_input import visualize_inputs
from diffusion_planner.utils.config import Config
import json
import numpy as np
import torch
from shutil import rmtree
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("args_json", type=Path)
    parser.add_argument("--save_path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    args_json = args.args_json
    save_path = args.save_path

    ext = input_path.suffix
    config_obj = Config(args_json)

    def process_one_data(input_path: Path, save_path: Path):
        loaded = np.load(input_path)
        data = {}
        for key, value in loaded.items():
            if key == "map_name" or key == "token":
                continue
            # add batch size axis
            data[key] = torch.tensor(np.expand_dims(value, axis=0))
        data = config_obj.observation_normalizer(data)

        visualize_inputs(data, config_obj.observation_normalizer, save_path)
        print(f"Saved to {save_path}")

    if ext == ".npz":
        process_one_data(input_path, save_path)
    elif ext == ".json":
        with open(input_path, "r") as f:
            path_list = json.load(f)
        path_list = sorted(path_list)
        save_path.mkdir(parents=True, exist_ok=True)
        assert save_path.is_dir()
        rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        sum = 0.0
        num = 0
        for path in path_list:
            path = Path(path)
            basename = path.stem
            curr_save_path = save_path / f"{basename}.png"
            start = time.time()
            process_one_data(Path(path), curr_save_path)
            end = time.time()
            elapsed_msec = (end - start) * 1000
            sum += elapsed_msec
            num += 1
            print(f"{sum / num:.2f} ms")
