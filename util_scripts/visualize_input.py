import pickle
import argparse
from pathlib import Path
from diffusion_planner.utils.visualize_input import visualize_inputs
from diffusion_planner.utils.config import Config
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pkl", type=Path)
    parser.add_argument("args_json", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_pkl = args.input_pkl
    args_json = args.args_json

    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    config_obj = Config(args_json)

    bs = -1
    for key, value in data.items():
        print(f"{key}\t{value.shape}")
        bs = value.shape[0] if bs == -1 else bs
        assert bs == value.shape[0], "Batch size mismatch"

    save_dir = Path("./visualize")
    save_dir.mkdir(exist_ok=True)
    for b in range(bs):
        inputs = deepcopy(data)
        for key, value in inputs.items():
            inputs[key] = value[b : b + 1]
        save_path = save_dir / f"input_{b:08d}.png"
        visualize_inputs(inputs, config_obj.observation_normalizer, save_path)
        print(f"Saved to {save_path}")
        break
