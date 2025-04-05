import pickle
import argparse
from pathlib import Path
from diffusion_planner.utils.visualize_input import visualize_inputs
from diffusion_planner.utils.config import Config


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

    for key, value in data.items():
        print(f"{key}\t{value.shape}")

    save_path = "./input.png"
    visualize_inputs(data, config_obj.observation_normalizer, save_path)
    print(f"Saved to {save_path}")
