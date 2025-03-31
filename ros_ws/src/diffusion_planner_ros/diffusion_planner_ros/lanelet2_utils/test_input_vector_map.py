from __future__ import annotations

from autoware_lanelet2_extension_python.projection import MGRSProjector
import lanelet2
import argparse
from lanelet_converter import convert_lanelet
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map_path",
        type=str,
        default="/home/shintarosakoda/data/misc/20250329_psim_rosbag/map/lanelet2_map.osm",
    )
    return parser.parse_args()


def test_projection():
    return MGRSProjector(lanelet2.io.Origin(0.0, 0.0))


def test_io(map_path, projection):
    return lanelet2.io.load(str(map_path), projection)


if __name__ == "__main__":
    args = parse_args()
    map_path = args.map_path

    result = convert_lanelet(map_path)
    print(f"{type(result)=}")

    ego_x = 3734.4
    ego_y = 73680.015625
    WIDTH = 50

    # Plot the map
    plt.figure()
    for segment_id, segment in result.lane_segments.items():
        waypoints = segment.polyline.waypoints
        print(f"{waypoints.shape=}")  # (N, 3)
        # x, yがegoからWIDTH内のものだけを抽出
        mask = (
            (waypoints[:, 0] > ego_x - WIDTH)
            & (waypoints[:, 0] < ego_x + WIDTH)
            & (waypoints[:, 1] > ego_y - WIDTH)
            & (waypoints[:, 1] < ego_y + WIDTH)
        )
        filtered_waypoints = waypoints[mask]
        plt.plot(filtered_waypoints[:, 0], filtered_waypoints[:, 1], "r-")

    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.axis("equal")
    save_path = "./test_input_vector_map.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved plot to {save_path}")
