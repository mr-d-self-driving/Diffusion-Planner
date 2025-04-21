export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone train_predictor.py \
--train_set_list "REPLACE_WITH_TRAIN_SET_LIST_PATH" \
