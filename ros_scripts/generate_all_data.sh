#!/bin/bash
set -eux

cd $(dirname $0)/../
set +eux
source ./install/setup.bash
set -eux

./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-12-11 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-01-24 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-02-04 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-03-25 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-04-16 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-04-30 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-05-07 &
./scripts/exec_parse_rosbag_for_one_day.sh /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2025-05-15 &

wait
