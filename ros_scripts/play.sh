#!/bin/bash

rate=${1:-1.0}

cd $(dirname $0)/../
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 bag play /home/shintarosakoda/data/nas_copy/tieriv_dataset/driving_dataset/bag/2025-04-16/10-47-50 -r $rate --remap /planning/scenario_planning/trajectory:=/unused
