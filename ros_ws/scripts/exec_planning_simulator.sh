#!/bin/bash
set -eux

# スクリプトが終了する際にすべてのバックグラウンドプロセスを停止するためのトラップを設定
trap "kill 0" EXIT

script_dir=$(readlink -f $(dirname $0))

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

python3 $script_dir/publish_initial_and_goal_from_yaml.py $HOME/work/Diffusion-Planner/ros_ws/local/2025-04-16_10-47-50_initialpose_and_goal.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $HOME/work/Diffusion-Planner/ros_ws/local/2025-04-16_11-01-45_initialpose_and_goal.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $HOME/work/Diffusion-Planner/ros_ws/local/2025-04-16_11-12-37_initialpose_and_goal.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $HOME/work/Diffusion-Planner/ros_ws/local/2025-04-16_11-37-42_initialpose_and_goal.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $HOME/work/Diffusion-Planner/ros_ws/local/2025-04-16_11-58-18_initialpose_and_goal.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $HOME/work/Diffusion-Planner/ros_ws/local/2025-04-16_12-16-03_initialpose_and_goal.yaml

# engage
ros2 topic pub /autoware/engage autoware_vehicle_msgs/msg/Engage "engage: true" -1
wait
