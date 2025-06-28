#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))
diffusion_planner_root=$(readlink -f $script_dir/..)
cd $diffusion_planner_root/../
rm -rf autoware
mkdir -p autoware/src
cd autoware/src
git clone https://github.com/autowarefoundation/autoware_cmake
git clone https://github.com/autowarefoundation/autoware_msgs
git clone https://github.com/autowarefoundation/autoware_lanelet2_extension
git clone https://github.com/astuff/pacmod3_msgs
ln -s $diffusion_planner_root ./
cd ../

rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
