#!/bin/bash
set -eux

target_dir=$(readlink -f $1)
cd $(dirname $0)

python3 ../diffusion_planner_ros/diffusion_planner_ros/conversion/torch2onnx.py \
    --config $target_dir/args.json \
    --ckpt $target_dir/best_model.pth \
    --onnx $target_dir/model.onnx \
    --sample_input_path /media/shintarosakoda/5EA85517A854EF51/20250416_onnx_work/sample_input.npz \
    --wrap_with_onnx_functions
