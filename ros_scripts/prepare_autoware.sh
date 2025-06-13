#!/bin/bash
set -eux

cd $(dirname $0)/../../
mkdir -p autoware/src
cd autoware/src
git clone https://github.com/autowarefoundation/autoware_cmake
git clone https://github.com/autowarefoundation/autoware_msgs
git clone https://github.com/autowarefoundation/autoware_lanelet2_extension
cd ../

rosdep update
rosdep install -y --from-paths src --ignore-src --rosdistro $ROS_DISTRO
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
