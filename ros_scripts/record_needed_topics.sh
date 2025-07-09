#!/bin/bash
set -eux

cd $(dirname $0)

SAVE_DIR=$(readlink -f $1)

ros2 bag record -o "$SAVE_DIR" \
  /localization/kinematic_state \
  /localization/acceleration \
  /perception/object_recognition/tracking/objects \
  /perception/traffic_light_recognition/traffic_signals \
  /planning/mission_planning/route \
  /vehicle/status/turn_indicators_status &

sleep 5

./publish_route.sh

wait
