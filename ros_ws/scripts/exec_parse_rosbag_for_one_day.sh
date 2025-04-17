#!/bin/bash
set -eu

cd $(dirname $0)
target_dir=$(readlink -f $1)

bag_dir_list=$(find $target_dir -mindepth 1 -maxdepth 1 -type d | sort)

for bag_dir in $bag_dir_list; do
    # bag_dir=/.../driving_dataset/bag/2024-07-18/10-10-58
    date=$(basename $(dirname $bag_dir))
    time=$(basename $bag_dir)

    # map_dir=/.../driving_dataset/map/2024-07-18
    # I assume that the map is consistent for the same day
    map_dir=$bag_dir/../../../map/$date
    map_path=$map_dir/lanelet2_map.osm

    # out_dir=/.../driving_dataset/preprocessed_for_diffusion_planner/2024-07-18/10-10-58
    out_dir=$bag_dir/../../../preprocessed_for_diffusion_planner/$date/$time

    if [ -d $out_dir ]; then
        echo "Already exists: $out_dir"
        continue
    fi

    echo $out_dir
    python3 ./parse_rosbag.py $bag_dir $map_path $out_dir --step 1
done
