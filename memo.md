# 環境構築

<https://github.com/SakodaShintaro/misc/tree/main/docker>

を使ってDockerコンテナを作成

```bash
# mediaの確認
ls /media

# CUDA 11.8のinstall
sudo apt purge -y --auto-remove "cuda*"
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# 環境変数の設定
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64" >> ~/.bashrc
source ~/.bashrc

# python3.9を入れられるように追加
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# python3.10削除
sudo apt remove -y --purge python3.10
sudo apt autoremove -y

# python3.9導入
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# 切り替え
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
sudo update-alternatives --config python3
python3 -V
sudo ln -s /usr/bin/python3.9 /usr/bin/python
python -V

# pipの導入
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
python3 -m pip install --upgrade pip
pip install --upgrade "pip<24.1"  # ダウングレードしないとomegaconfが上手く入らない
# ref. https://github.com/OFA-Sys/ONE-PEACE/issues/55

# nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r requirements.txt

# Diffusion-Planner
cd ~/work/Diffusion-Planner/
pip install -e .
# バージョンの兼ね合いにより分けて入れる必要がある
pip install pytorch_lightning==2.0.1 tensorboard==2.11.2 timm==1.0.10 mmengine wandb
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# torchの確認
python3 -c "import torch"
```

## ROS2 humble導入

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-humble-ros-base
```

## Tip: 偶数行だけ残す

```bash
#!/bin/bash
awk 'NR % 2 == 1' diffusion_planner_training.json > even_lines.json
```
