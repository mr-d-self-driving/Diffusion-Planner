from __future__ import annotations

from autoware_lanelet2_extension_python.projection import MGRSProjector
import lanelet2
import argparse
from pathlib import Path
from lanelet_converter import convert_lanelet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_path", type=Path)
    return parser.parse_args()


def test_projection():
    return MGRSProjector(lanelet2.io.Origin(0.0, 0.0))


def test_io(map_path, projection):
    return lanelet2.io.load(str(map_path), projection)


if __name__ == "__main__":
    args = parse_args()
    map_path = args.map_path
    map_path = str(map_path)

    result = test_io(map_path, test_projection())
    print(f"{type(result)=}")

    result = convert_lanelet(map_path)
