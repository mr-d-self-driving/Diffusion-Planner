#!/bin/bash

cd $(dirname $0)/../
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 run diffusion_planner_ros diffusion_planner_node --ros-args \
    -p vector_map_path:=/home/shintarosakoda/data/driving_data/map/lanelet2_map.osm \
    -p config_json_path:=/media/shintarosakoda/5EA85517A854EF51/diffusion_planner_training_result/train_result/2025-03-20-180651_datasize_1M/args.json \
    -p ckpt_path:=/media/shintarosakoda/5EA85517A854EF51/diffusion_planner_training_result/train_result/2025-03-20-180651_datasize_1M/latest.pth \
    -p use_sim_time:=true \
    -p batch_size:=1 \
        -p backend:=PYTHORCH \
    -p onnx_path:=/home/danielsanchez/Diffusion-Planner/ros_ws/src/diffusion_planner_ros/diffusion_planner_ros/conversion/model.onnx


