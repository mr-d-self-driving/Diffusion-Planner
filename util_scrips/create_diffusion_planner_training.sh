#!/bin/bash
set -eux

# This script convert the dataset to a list.
# $ ls -1
# us-nv-las-vegas-strip_150e044bab0a57a8.npz
# us-nv-las-vegas-strip_2324ed8cd5a75cb9.npz
# us-nv-las-vegas-strip_47ceedefd88158a3.npz
# us-nv-las-vegas-strip_4e507af0fcf75ad3.npz
# us-nv-las-vegas-strip_60b40cbd09e85a06.npz
# ...
# ->
# [
#     "us-nv-las-vegas-strip_150e044bab0a57a8.npz",
#     "us-nv-las-vegas-strip_2324ed8cd5a75cb9.npz",
#     "us-nv-las-vegas-strip_47ceedefd88158a3.npz",
#     "us-nv-las-vegas-strip_4e507af0fcf75ad3.npz",
#     "us-nv-las-vegas-strip_60b40cbd09e85a06.npz",
#     ...
# ]

target_dir=$1
output_file="./diffusion_planner_training.json"

body=$(ls $target_dir | sed 's/^/    "/g' | sed 's/$/"/g' | sed 's/$/,/g' | sed '$ s/.$//')

echo "[" > $output_file
echo "$body" >> $output_file
echo "]" >> $output_file
