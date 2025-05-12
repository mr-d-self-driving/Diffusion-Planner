#!/bin/bash
set -eu

target_dir=$(readlink -f $1)
cd $(dirname $0)

bag_dir_list=$(find $target_dir -mindepth 1 -maxdepth 1 -type d | sort)

date=$(basename $target_dir)
save_root=$target_dir/../../../../private_workspace/diffusion_planner/preprocessed_for_diffusion_planner7/$date
mkdir -p $save_root
save_root=$(readlink -f $save_root)
echo $save_root

for bag_dir in $bag_dir_list; do
    # bag_dir=/.../driving_dataset/bag/2024-07-18/10-10-58
    time=$(basename $bag_dir)

    # map_dir=/.../driving_dataset/map/2024-07-18
    # I assume that the map is consistent for the same day
    map_dir=$bag_dir/../../../map/$date
    map_path=$map_dir/lanelet2_map.osm

    # if there is map/$date/$time, use it
    if [ -d $map_dir/$time ]; then
        map_path=$map_dir/$time/lanelet2_map.osm
    fi

    # out_dir=$save_root/10-10-58
    out_dir=$save_root/$time

    if [ -d $out_dir ]; then
        echo "Already exists: $out_dir"
        continue
    fi

    echo $out_dir
    python3 ./parse_rosbag.py $bag_dir $map_path $out_dir --step 1 --log_dir $save_root
done

zip -r $save_root.zip $save_root
