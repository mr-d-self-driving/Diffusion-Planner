#!/bin/bash
set -eux
script_dir=$(readlink -f $(dirname $0))
python3 $script_dir/publish_initial_and_goal_from_yaml.py $script_dir/route/route01_teleport_to_miraikan.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $script_dir/route/route02_miraikan_to_teleport.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $script_dir/route/route03_teleport_to_kokusaitenjijo.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $script_dir/route/route04_kokusaitenjijo_to_miraikan.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $script_dir/route/route05_miraikan_to_kokusaitenjijo.yaml
# python3 $script_dir/publish_initial_and_goal_from_yaml.py $script_dir/route/route06_kokusaitenjijo_to_teleport.yaml
