#!/bin/bash
set -eux

cd $(dirname $0)

data_root0=/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_for_diffusion_planner12
data_root1=/mnt/nvme1/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_for_diffusion_planner12
data_root2=/mnt/nvme2/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_for_diffusion_planner12
data_root_aux=/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_psim

python3 ../diffusion_planner/util_scripts/create_train_set_path.py \
    $data_root0/2024-07-18 \
    $data_root0/2024-12-11 \
    $data_root0/2025-01-24 \
    $data_root0/2025-02-04 \
    $data_root0/2025-03-25 \
    $data_root0/2025-04-16 \
    $data_root1/2025-04-30 \
    $data_root1/2025-05-07 \
    $data_root_aux/2025-06-10 \
    --save_path $data_root0/train_set_path_with_psim20250610.json
