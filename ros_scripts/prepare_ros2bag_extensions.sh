#!/bin/bash
set -eux

script_dir=$(readlink -f $(dirname $0))

cd $script_dir
# create workspace for extension
rm -rf ./extension_ws
mkdir -p ./extension_ws/src
# clone extension package
cd ./extension_ws/src
git clone https://github.com/tier4/ros2bag_extensions
# build workspace
cd $script_dir/extension_ws
set +eux
source /opt/ros/humble/setup.bash
set -eux
rosdep install --from-paths . --ignore-src --rosdistro=${ROS_DISTRO}
colcon build --symlink-install --catkin-skip-building-tests --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
