from __future__ import annotations

import argparse
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import (
    convert_lanelet,
    get_input_feature,
)
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_path",
        type=str,
        default="/home/shintarosakoda/data/misc/20250329_psim_rosbag/map/lanelet2_map.osm",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    map_path = args.map_path

    result = convert_lanelet(map_path)
    print(f"{type(result)=}")

    ego_x = 3734.4
    ego_y = 73680.015625
    ego_z = 19.519
    ego_qx = 0.0005874986553164094
    ego_qy = -0.0022261576737295824
    ego_qz = 0.2551699815886937
    ego_qw = 0.9668934685700217
    RANGE = 100

    waypoints_list = get_input_feature(result, ego_x, ego_y, ego_z, ego_qx, ego_qy, ego_qz, ego_qw, RANGE)
    print(f"{len(waypoints_list)=}")

    plt.figure()
    for i in range(len(waypoints_list)):
        filtered_waypoints = waypoints_list[i]
        plt.plot(filtered_waypoints[:, 0], filtered_waypoints[:, 1], "r-")

    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.xlim(-RANGE, RANGE)
    plt.ylim(-RANGE, RANGE)
    save_path = "./test_input_vector_map.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved plot to {save_path}")
