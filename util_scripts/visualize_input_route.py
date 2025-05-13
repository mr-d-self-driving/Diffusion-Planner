import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_path = args.json_path

    save_dir = json_path.parent / f"{json_path.stem}_visualize"
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(json_path, "r") as f:
        path_list = json.load(f)
    path_list = sorted(path_list)

    pos_x = []
    pos_y = []
    prev_time = -1
    prev_route_id = -1

    for i, path in tqdm(enumerate(path_list)):
        print(path)
        pose_json = path.replace(".npz", ".json")
        path = Path(path)
        path_id = int(path.stem.split("_")[1])
        route_id = path_id // int(1e8)
        path_id = path_id % int(1e8)
        date = path.parent.parent.name
        time = path.parent.name

        if (route_id != prev_route_id or time != prev_time) and (i > 0):
            plt.title(f"{date}/{path.stem}")
            plt.scatter(pos_x, pos_y)
            s = plt.text(pos_x[0], pos_y[0], "S", fontsize=12)
            g = plt.text(pos_x[-1], pos_y[-1], "G", fontsize=12)
            plt.axis("equal")
            plt.savefig(save_dir / f"{date}_{path.stem}.png")
            pos_x = []
            pos_y = []
            s.remove()
            g.remove()

        with open(pose_json, "r") as f:
            pose = json.load(f)
        pos_x.append(pose["x"])
        pos_y.append(pose["y"])
        prev_route_id = route_id
        prev_time = time
