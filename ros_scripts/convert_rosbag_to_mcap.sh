#!/bin/bash
set -eux

TARGET_DIR=$(readlink -f $1)
# metadata.yamlを探すので、TARGET_DIRに与えるのは
# /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18/10-05-28
# でも
# /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18/
# でも
# /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/
# でも良い

set +eux
source ~/pilot-auto.xx1/install/setup.bash
set -eux

metadata_yaml_list=$(find $TARGET_DIR -name "metadata.yaml" | sort)

for metadata_yaml in $metadata_yaml_list; do
    curr_target_dir=$(dirname $metadata_yaml)  # 例) /mnt/nvme0/sakoda/nas_copy/tieriv_dataset/driving_dataset/bag/2024-07-18/10-05-28
    time=$(basename $curr_target_dir)  # 10-05-28
    date=$(basename $(dirname $curr_target_dir))  # 2024-07-18

    bag_mcap_dir=$(readlink -f ${curr_target_dir}/../../../bag_mcap)
    if [ -d ${bag_mcap_dir}/${date}/${time} ]; then
        echo "Skipping already converted bag: ${bag_mcap_dir}/${date}/${time}"
        continue
    fi
    mkdir -p ${bag_mcap_dir}/${date}

    echo "
    output_bags:
    - uri: /${bag_mcap_dir}/${date}/${time}
      storage_id: mcap
      all: true
    " > config.yaml

    ros2 bag convert -i $curr_target_dir -o config.yaml

    rm config.yaml
done
