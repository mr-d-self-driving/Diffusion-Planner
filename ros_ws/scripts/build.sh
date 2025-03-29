#!/bin/bash

cd $(dirname $0)/../
colcon build \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  --continue-on-error \
  --packages-up-to diffusion_planner_ros
