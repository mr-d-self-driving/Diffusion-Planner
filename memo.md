# 環境構築

<https://github.com/SakodaShintaro/misc/tree/main/docker>

を使ってDockerコンテナを作成

```bash
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
```
