export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

###################################
# User Configuration Section
###################################

# Set training data path
TRAIN_SET_LIST_PATH="REPLACE_WITH_TRAIN_SET_LIST_PATH"
MODEL_PATH="REPLACE_WITH_MODEL_PATH"
ARGS_JSON_PATH="REPLACE_WITH_MODEL_PATH"
###################################

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone valid_predictor.py \
--train_set_list  $TRAIN_SET_LIST_PATH \
--resume_model_path $MODEL_PATH \
--args_json_path $ARGS_JSON_PATH \
