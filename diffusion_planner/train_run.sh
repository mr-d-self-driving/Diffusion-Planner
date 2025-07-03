#!/bin/bash
set -ux
cd $(dirname $0)

exp_name=${1}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

rm -f /tmp/tmp_dist_init

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--exp_name ${exp_name} \
--train_set_list "/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver16_realdata/path_list_train.json" \
--valid_set_list "/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_ver16_realdata/path_list_valid.json" \
--use_wandb True \
--diffusion_model_type "x_start" \
2>&1 | tee logs/result_$(date +%Y%m%d_%H%M%S).txt

save_dir_name=$(ls ./training_log/diffusion-planner-training/ | tail -n 1)
./valid_run.sh ${save_dir_name}
