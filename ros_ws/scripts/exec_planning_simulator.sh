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
  map_path:=$HOME/data/nas_copy/tieriv_dataset/driving_dataset/map/2025-04-16/ \
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
    position: { x: 89423.95818648902, y: 43242.31510616808, z: 5.887557728617264 },
    orientation: { x: -0.0017407279235030458, y: -0.005716426991434616, z: -0.482224866284304, w: 0.8760270947098802 }
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
ros2 topic pub -1 /planning/mission_planning/goal geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 181, nanosec: 289995947},
frame_id: 'map'},
pose: {position: {x: 89153.3642, y: 42417.9992, z: 7.0153 },
    orientation: {x: 0.0, y: 0.0, z: 0.9289624344790391, w: -0.3701740068221657 }
}
}'
sleep 1

# engage
ros2 topic pub /autoware/engage autoware_vehicle_msgs/msg/Engage "engage: true" -1
wait
