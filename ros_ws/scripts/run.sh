#!/bin/bash

cd $(dirname $0)/../
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 run diffusion_planner_ros diffusion_planner_node --ros-args \
    -p vector_map_path:=/home/danielsanchez/diffusion_planner_data/lanelet2_map.osm \
    -p config_json_path:=/home/danielsanchez/diffusion_planner_data/args.json \
    -p ckpt_path:=/home/danielsanchez/diffusion_planner_data/latest.pth \
    -p use_sim_time:=true \
    -p batch_size:=1 \
    -p backend:=PYTHORCH \
    -p onnx_path:=/home/danielsanchez/Diffusion-Planner/ros_ws/src/diffusion_planner_ros/diffusion_planner_ros/conversion/model.onnx


