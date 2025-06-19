# 1. Setup

## (Optional) create venv

```bash
sudo apt install python3-pip -y
sudo apt install python3-venv -y
python3 -m venv .venv
source ./.venv/bin/activate
```

## Install libraries

```bash
# nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .

# Diffusion-Planner
cd ~/work/Diffusion-Planner/diffusion_planner
python -m pip install pip==24.1

pip install -r requirements_nuplan-devkit_fixed.txt
pip install -r requirements.txt
pip install -e .

# check torch
python3 -c "import torch; print(torch.cuda.is_available())"

# install ros-humble
./ros_scripts/download_ros-hubmle.sh
```

# 2. Create dataset

## 2.1. Prepare rosbags

We assume the following directory structure:

```bash
driving_dataset$ tree . -L 2
.
├── bag
│   ├── 2024-07-18
│   │ ├── 10-05-28
│   │ ├── 10-05-51
│   │ ├── ...
│   │ ├── 16-10-07
│   │ └── 16-27-15
│   ├── 2024-12-11
│   ├── 2025-01-24
│   ├── 2025-02-04
│   ├── 2025-03-25
│   └── 2025-04-16
└── map
     ├── 2024-07-18
     │   ├── lanelet2_map.osm
     │   ├── pointcloud_map_metadata.yaml
     │   ├── pointcloud_map.pcd
     │   └── stop_points.csv
     ├── 2024-12-11
     ├── 2025-01-24
     ├── 2025-02-04
     ├── 2025-03-25
     └── 2025-04-16
```

## 2.2. Convert to diffusion_planner's format (npz)

```bash
./ros_scripts/generate_all_data.sh
```

or use `parse_rosbag_for_directory.py` directly.

```bash
python3 ./parse_rosbag_for_directory.py <rosbag_path> <vectormap_path> <etc>
```

# 3. Train

Edit `train_run.sh` and run

```bash
cd ./diffusion_planner
./train_run.sh
```
