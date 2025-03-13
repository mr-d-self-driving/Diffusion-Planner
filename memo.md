# 環境構築

<https://github.com/SakodaShintaro/misc/tree/main/docker>

を使ってDockerコンテナを作成

```bash
# mediaの確認
ls /media

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
pip install -r requirements_torch.txt

# torchの確認
python3 -c "import torch"

sudo apt purge -y --auto-remove "cuda*"

# CUDA 11.8のリポジトリを追加
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# CUDA 11.8をインストール
sudo apt install -y cuda-toolkit-11-8

# 環境変数の設定
pip uninstall torch torchvision -y
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64

python3 -c "import torch"
```
