#!/bin/bash
set -eux

cd $(dirname $0)

set +eux
source ~/pilot-auto.xx1/install/setup.bash
set -eux

python3 ./parse_rosbag_for_directory.py \
    /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/ \
    /mnt/nvme1/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/ \
    /mnt/nvme2/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/ \
    /mnt/nvme3/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/ \
    --save_root /mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver15_realdata \
    --step 1 \
    --limit -1 \
    --min_frames 1800 \
    --search_nearest_route 1
