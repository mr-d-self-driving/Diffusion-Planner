#!/bin/bash
set -eux

cd $(dirname $0)

data_root=/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver16_realdata

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root/2024-07-18 \
    $data_root/2024-12-11 \
    $data_root/2025-01-24 \
    $data_root/2025-02-04 \
    $data_root/2025-03-25 \
    $data_root/2025-04-16 \
    $data_root/2025-04-30 \
    $data_root/2025-05-07 \
    $data_root/2025-05-15 \
    $data_root/2025-05-22 \
    $data_root/2025-05-29 \
    $data_root/2025-06-09 \
    --save_path $data_root/path_list_train.json

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root/2025-06-12 \
    $data_root/2025-06-16 \
    --save_path $data_root/path_list_valid.json
