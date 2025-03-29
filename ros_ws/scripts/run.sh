#!/bin/bash

cd $(dirname $0)/../
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 run diffusion_planner_ros diffusion_planner_node
