from __future__ import annotations

import argparse
from diffusion_planner_ros.lanelet2_utils.lanelet_converter import convert_lanelet
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np


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
    WIDTH = 100

    # 自車中心に座標変換するための行列を作成
    rot = Rotation.from_quat([ego_qx, ego_qy, ego_qz, ego_qw])
    translation = np.array([ego_x, ego_y, ego_z])
    transform_matrix = rot.as_matrix()
    transform_matrix_4x4 = np.eye(4)
    transform_matrix_4x4[:3, :3] = transform_matrix
    transform_matrix_4x4[:3, 3] = translation
    inv_transform_matrix_4x4 = np.eye(4)
    inv_transform_matrix_4x4[:3, :3] = transform_matrix.T
    inv_transform_matrix_4x4[:3, 3] = -transform_matrix.T @ translation

    # Plot the map
    plt.figure()
    for segment_id, segment in result.lane_segments.items():
        waypoints = segment.polyline.waypoints
        print(f"{waypoints.shape=}")  # (N, 3)
        # 自車座標系に変換
        waypoints_4xN = np.vstack((waypoints.T, np.ones(waypoints.shape[0])))
        waypoints_ego = inv_transform_matrix_4x4 @ waypoints_4xN
        waypoints = waypoints_ego[:3, :].T

        # x, yがegoからWIDTH内のものだけを抽出
        mask = (
            (waypoints[:, 0] > -WIDTH)
            & (waypoints[:, 0] < WIDTH)
            & (waypoints[:, 1] > -WIDTH)
            & (waypoints[:, 1] < WIDTH)
        )
        filtered_waypoints = waypoints[mask]
        plt.plot(filtered_waypoints[:, 0], filtered_waypoints[:, 1], "r-")

    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.xlim(-WIDTH, WIDTH)
    plt.ylim(-WIDTH, WIDTH)
    save_path = "./test_input_vector_map.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved plot to {save_path}")
