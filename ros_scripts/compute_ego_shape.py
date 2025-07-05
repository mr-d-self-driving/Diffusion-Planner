"""Compute the ego shape from the information of vehicle_info.param.yaml"""

import argparse
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("vehicle_info_path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vehicle_info_path = args.vehicle_info_path

    with open(vehicle_info_path, "r") as f:
        vehicle_info = yaml.safe_load(f)

    vehicle_params = vehicle_info["/**"]["ros__parameters"]
    wheel_base = vehicle_params["wheel_base"]
    print(f"{wheel_base=}")

    ego_length = vehicle_params["front_overhang"] + wheel_base + vehicle_params["rear_overhang"]
    print(f"{ego_length=}")

    ego_width = (
        vehicle_params["wheel_tread"]
        + vehicle_params["left_overhang"]
        + vehicle_params["right_overhang"]
    )
    print(f"{ego_width=}")
