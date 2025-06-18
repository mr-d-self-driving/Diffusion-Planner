import argparse
from pathlib import Path

from parse_rosbag import main as parse_rosbag_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="+")
    parser.add_argument("--save_root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--min_frames", type=int, default=1700)
    parser.add_argument("--search_nearest_route", type=int, default=1)
    args = parser.parse_args()
    target_dir_list = args.target_dir_list
    save_root = args.save_root
    step = args.step
    limit = args.limit
    min_frames = args.min_frames
    search_nearest_route = args.search_nearest_route

    save_root = save_root.resolve()

    # search "metadata.yaml"
    metadata_list = []
    for target_dir in target_dir_list:
        metadata_list.extend(list(target_dir.glob("**/metadata.yaml")))
    bag_dir_list = [
        metadata_path.parent for metadata_path in metadata_list if metadata_path.is_file()
    ]

    for bag_path in bag_dir_list:
        date = bag_path.parent.name
        time = bag_path.name

        map_dir = bag_path.parent.parent.parent / "map" / date
        vector_map_path = map_dir / "lanelet2_map.osm"

        # if there is map/$date/$time, use it
        if (map_dir / time).is_dir():
            vector_map_path = map_dir / time / "lanelet2_map.osm"

        (save_root / date).mkdir(parents=True, exist_ok=True)
        save_dir = (save_root / date / time).resolve()

        if save_dir.is_dir():
            print(f"Already exists: {save_dir}")
            continue

        parse_rosbag_main(
            rosbag_path=bag_path,
            vector_map_path=vector_map_path,
            save_dir=save_dir,
            step=step,
            limit=limit,
            min_frames=min_frames,
            search_nearest_route=search_nearest_route,
        )
