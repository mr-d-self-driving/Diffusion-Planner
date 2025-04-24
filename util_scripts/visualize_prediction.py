import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import torch
from parse_prediction_results import calc_loss
from tqdm import tqdm

from diffusion_planner.utils.config import Config
from diffusion_planner.utils.visualize_input import visualize_inputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--args_json", type=Path, required=True)
    parser.add_argument("--valid_data_list", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions_dir = args.predictions_dir
    args_json = args.args_json
    valid_data_list = args.valid_data_list
    save_dir = args.save_dir

    config_obj = Config(args_json)

    with open(valid_data_list, "r") as f:
        valid_data_path_list = json.load(f)

    prediction_path_list = sorted(predictions_dir.glob("**/*.npz"))

    assert len(prediction_path_list) == len(valid_data_path_list)

    save_dir.mkdir(parents=True, exist_ok=True)
    assert save_dir.is_dir()
    rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def process_one_pair(pair):
        valid_data_path, prediction_path = pair
        valid_data_path = Path(valid_data_path)
        prediction_path = Path(prediction_path)
        valid_data = np.load(valid_data_path)
        prediction = np.load(prediction_path)

        valid_data_dict = {}
        for key, value in valid_data.items():
            if key == "map_name" or key == "token":
                continue
            # add batch size axis
            valid_data_dict[key] = torch.tensor(np.expand_dims(value, axis=0))
        valid_data_dict = config_obj.observation_normalizer(valid_data_dict)

        prediction = prediction["prediction"]  # (1 + P, T, D)
        loss_ego, loss_nei, neighbors_future_valid = calc_loss(valid_data, prediction)
        # loss_ego (T, 4)
        # loss_nei (P, T, 4)
        loss_ego = np.sqrt(loss_ego)
        loss_nei = np.sqrt(loss_nei)
        loss_ego_mean = np.mean(loss_ego)

        fig, ax = visualize_inputs(valid_data_dict, config_obj.observation_normalizer)

        # plot prediction
        ax.plot(
            prediction[0, :, 0],
            prediction[0, :, 1],
            color="orange",
            label="prediction",
            linewidth=2,
        )
        # 3sec, 5sec, 8sec
        title = f"loss_mean={loss_ego_mean:.2f}"
        for timestep in [30, 50, 80]:
            index = timestep - 1
            diff_m = np.sqrt(loss_ego[index, 0] ** 2 + loss_ego[index, 1] ** 2)
            ax.text(prediction[0, index, 0], prediction[0, index, 1], f"{diff_m:.2f}[m]")
            ax.plot(prediction[0, index, 0], prediction[0, index, 1], color="black", marker="x")
            title += f", loss{timestep // 10}sec={diff_m:.2f}[m]"

        ax.set_title(title)

        plt.savefig(save_dir / f"{valid_data_path.stem}.png")
        plt.close()

        loss_dict = {
            "loss_ego_mean": loss_ego_mean,
        }
        json.dump(
            loss_dict,
            open(save_dir / f"{valid_data_path.stem}.json", "w"),
            indent=4,
        )

    pool = Pool(16)
    with tqdm(total=len(valid_data_path_list)) as pbar:
        for _ in pool.imap_unordered(
            process_one_pair, zip(valid_data_path_list, prediction_path_list)
        ):
            pbar.update(1)
