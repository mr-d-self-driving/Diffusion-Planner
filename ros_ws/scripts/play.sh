#!/bin/bash

rate=${1:-2.0}

cd $(dirname $0)/../
source /opt/ros/humble/setup.bash
source ./install/setup.bash
ros2 bag play /home/shintarosakoda/data/misc/20250329_psim_rosbag/rosbag2_2025_03_29-13_12_14 -r $rate
