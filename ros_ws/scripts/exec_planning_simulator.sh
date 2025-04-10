#!/bin/bash
set -eux

# スクリプトが終了する際にすべてのバックグラウンドプロセスを停止するためのトラップを設定
trap "kill 0" EXIT

# 読み込み
set +eux
source ./install/setup.bash
set -eux

# planning_simulator
ros2 launch autoware_launch planning_simulator.launch.xml \
  map_path:=$HOME/data/driving_data/map/ \
  vehicle_model:=taxi sensor_model:=aip_xx1 &

# 立ち上がるまで待つ
while ! ros2 service type /planning/scenario_planning/scenario_selector/get_parameters; do
  echo "waiting for /planning/scenario_planning/scenario_selector/get_parameters"
  sleep 4
done

# 安定性のためにさらに少し待つ
sleep 10

# initialpose
# ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{
#   header: { frame_id: "map" },
#   pose: {
#     pose: {
#       position: { x: 86127.49944449718, y: 42993.304310392974, z: 2.7531935109593793 },
#       orientation: { x: 0.019454575063457234, y: 0.004655547702116446, z: -0.9667914651761493, w: 0.25478247240979535 }
#     },
#     covariance: [0.25, 0.0,  0.0, 0.0, 0.0, 0.0,
#                  0.0,  0.25, 0.0, 0.0, 0.0, 0.0,
#                  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
#                  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
#                  0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
#                  0.0,  0.0,  0.0, 0.0, 0.0, 0.06853891909122467]
#   }
# }'
ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped '{
  header: { frame_id: "map" },
  pose: {
    pose: {
      position: { x: 86119.53125, y: 42994.8984375, z: 0.0 },
      orientation: { x: 0.0, y: 0.0, z: -0.4738380103640769, w: 0.8806120257719701 }
    },
    covariance: [0.25, 0.0,  0.0, 0.0, 0.0, 0.0,
                 0.0,  0.25, 0.0, 0.0, 0.0, 0.0,
                 0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                 0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                 0.0,  0.0,  0.0, 0.0, 0.0, 0.0,
                 0.0,  0.0,  0.0, 0.0, 0.0, 0.06853891909122467]
  }
}'
sleep 1

# goal
ros2 topic pub -1 /planning/mission_planning/goal geometry_msgs/msg/PoseStamped '{
  header: {
    stamp: {sec: 181, nanosec: 289995947},
    frame_id: 'map'},
  pose: {
    position: { x: 89410.0238, y: 43213.9237, z: 5.738 },
    orientation: { x: 0.0, y: 0.0, z: 0.867423225594017, w: 0.49757104789172696 }
  }
}'
sleep 1

# engage
ros2 topic pub /autoware/engage autoware_vehicle_msgs/msg/Engage "engage: true" -1
wait
