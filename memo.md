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

# nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .

# Diffusion-Planner
cd ~/work/Diffusion-Planner/
python -m pip install pip==24.1
pip install -r requirements_nuplan-devkit_fixed_opencv.txt
pip install --upgrade setuptools
pip install -e .
# バージョンの兼ね合いにより分けて入れる必要がある
pip install pytorch_lightning==2.0.1 tensorboard==2.11.2 timm==1.0.10 mmengine wandb
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# torchの確認
python3 -c "import torch"

# rosの導入
./util_scripts/download_ros-hubmle.sh
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
```
