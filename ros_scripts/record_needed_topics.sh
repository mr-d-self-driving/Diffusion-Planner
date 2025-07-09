#!/bin/bash
set -eux

cd $(dirname $0)

SAVE_DIR=$(readlink -f $1)

rm -rf "$SAVE_DIR"

ros2 bag record -o "$SAVE_DIR" \
  /localization/kinematic_state \
  /localization/acceleration \
  /perception/object_recognition/tracking/objects \
  /perception/traffic_light_recognition/traffic_signals \
  /planning/mission_planning/route \
  /vehicle/status/turn_indicators_status &

sleep 5

python3 /home/shintarosakoda/pilot-auto.xx1_dp/src/Diffusion-Planner/ros_scripts/publish_traffic_light.py &

./publish_route.sh

wait
