#!/bin/bash
set -eux

model_dir=$(readlink -f $1)
cd $(dirname $0)/../

set +eux
source /opt/ros/humble/setup.bash
source ../../install/setup.bash
set -eux

ros2 run diffusion_planner_ros diffusion_planner_node --ros-args \
    -p vector_map_path:=$HOME/data/nas_copy/tieriv_dataset/driving_dataset/map/2025-04-16/lanelet2_map.osm \
    -p config_json_path:=$model_dir/args.json \
    -p ckpt_path:=$model_dir/latest.pth \
    -p onnx_path:=$model_dir/model.onnx \
    -p backend:=ONNXRUNTIME \
    -p batch_size:=1 \
    -p use_sim_time:=true \
