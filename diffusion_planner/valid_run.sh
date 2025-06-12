#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=0

###################################
# User Configuration Section
###################################

# Set training data path
DIR_NAME=${1}
MODEL_DIR="./training_log/diffusion-planner-training/$DIR_NAME/"
VALID_SET_LIST_PATH="/mnt/nvme0/sakoda/nas_copy/private_workspace/diffusion_planner/preprocessed_for_diffusion_planner12/valid_set_teleport_3routes_path.json"
MODEL_PATH="$MODEL_DIR/latest.pth"
ARGS_JSON_PATH="$MODEL_DIR/args.json"
SAVE_DIR=/mnt/nvme0/sakoda/${DIR_NAME}/predictions

rm -f /tmp/tmp_dist_init

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone valid_predictor.py \
--valid_set_list  $VALID_SET_LIST_PATH \
--resume_model_path $MODEL_PATH \
--args_json_path $ARGS_JSON_PATH \
--save_predictions_dir $SAVE_DIR \

python3 util_scripts/visualize_prediction.py \
  --predictions_dir $SAVE_DIR \
  --args_json $ARGS_JSON_PATH \
  --valid_data_list $VALID_SET_LIST_PATH
