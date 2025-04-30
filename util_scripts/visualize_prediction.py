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
    parser.add_argument("--save_dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictions_dir = args.predictions_dir
    args_json = args.args_json
    valid_data_list = args.valid_data_list
    save_dir = args.save_dir

    if save_dir is None:
        save_dir = predictions_dir.parent / f"visualization"

    config_obj = Config(args_json)

    with open(valid_data_list, "r") as f:
        valid_data_path_list = json.load(f)

    prediction_path_list = sorted(predictions_dir.glob("**/*.npz"))
    loss_path_list = sorted(predictions_dir.glob("**/*.json"))

    info_path_list = [
        Path(valid_data_path).parent / f"{Path(valid_data_path).stem}.json"
        for valid_data_path in valid_data_path_list
    ]
    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")
    trajectory_x = []
    trajectory_y = []
    loss_3sec = []
    for info_path, loss_path in zip(info_path_list, loss_path_list):
        assert info_path.is_file()
        pose_data = json.load(open(info_path, "r"))
        x = pose_data["x"]
        trajectory_x.append(x)
        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        y = pose_data["y"]
        trajectory_y.append(y)
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))

        loss_data = json.load(open(loss_path, "r"))
        loss_3sec.append(loss_data["loss_ego_3sec"])
    lx = max_x - min_x
    ly = max_y - min_y
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    l_max = max(lx, ly) * 1.2

    assert len(prediction_path_list) == len(valid_data_path_list)

    save_dir.mkdir(parents=True, exist_ok=True)
    assert save_dir.is_dir()
    rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def process_one_pair(pair):
        valid_data_path, prediction_path = pair
        valid_data_path = Path(valid_data_path)
        prediction_path = Path(prediction_path)
        info_data_path = valid_data_path.parent / f"{valid_data_path.stem}.json"
        valid_data = np.load(valid_data_path)
        prediction = np.load(prediction_path)
        info_data = json.load(open(info_data_path, "r"))
        ego_x = info_data["x"]
        ego_y = info_data["y"]

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

        fig, ax = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [2, 1]})
        visualize_inputs(valid_data_dict, config_obj.observation_normalizer, save_path=None, ax=ax[0])

        # plot prediction
        # Ego
        ax[0].plot(
            prediction[0, :, 0],
            prediction[0, :, 1],
            color="orange",
            label="prediction",
            linewidth=2,
        )
        # 3sec, 5sec, 8sec
        title = f""
        for timestep in [30, 50, 80]:
            index = timestep - 1
            diff_m = np.sqrt(loss_ego[index, 0] ** 2 + loss_ego[index, 1] ** 2)
            ax[0].plot(prediction[0, index, 0], prediction[0, index, 1], color="black", marker="x")
            if timestep == 30:
                title += f"loss{timestep // 10}sec={diff_m:.2f}[m]\n"

        # Neighbors
        for i in range(prediction.shape[0] - 1):
            ax[0].plot(
                prediction[i + 1, :, 0],
                prediction[i + 1, :, 1],
                color="teal",
                alpha=0.5,
            )

        ax[0].set_title(title)

        ax[1].scatter(
            trajectory_x,
            trajectory_y,
            c=loss_3sec,
            marker="o",
            s=10,
        )
        ax[1].scatter(
            ego_x,
            ego_y,
            color="red",
            marker="+",
            s=50,
        )
        ax[1].set_xlabel("x[m]")
        ax[1].set_ylabel("y[m]")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].grid(True)
        ax[1].set_title("loss3sec")
        ax[1].set_aspect("equal")

        plt.colorbar(ax[1].collections[0], ax=ax[1])

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
