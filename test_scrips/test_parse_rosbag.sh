#!/bin/bash
set -eux

cd $(dirname $0)/..

result_dir=/mnt/nvme0/sakoda/test/$(date +%Y%m%d_%H%M%S)_test_parse_rosbag

rm -rf ${result_dir}
mkdir -p ${result_dir}/tmp

python3 ./ros_scripts/parse_rosbag.py \
    /mnt/nvme3/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-06-12/10-19-35 \
    /mnt/nvme3/sakoda/nas_copy/tieriv_dataset/driving_dataset/map/2025-06-12/10-19-35/lanelet2_map.osm \
    ${result_dir}/tmp \
    --limit 30000000 \
    --min_frames 0 2>&1 | tee $result_dir/result_$(date +%Y%m%d_%H%M%S).txt

python3 ./diffusion_planner/util_scripts/create_train_set_path.py ${result_dir}/tmp

python3 ./diffusion_planner/util_scripts/visualize_input.py ${result_dir}/path_list.json \
    /home/ubuntu/sakoda/Diffusion-Planner_npf/diffusion_planner/training_log/diffusion-planner-training/20250611-145217_with_psim_20250610/args.json \
    --save_path ${result_dir}/visualize_result

~/misc/ffmpeg_lib/make_mp4_from_unsequential_png.sh ${result_dir}/visualize_result
