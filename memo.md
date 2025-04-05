# 環境構築

<https://github.com/SakodaShintaro/misc/tree/main/docker>

を使ってDockerコンテナを作成

```bash
# nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .

# Diffusion-Planner
cd ~/work/Diffusion-Planner/
python -m pip install pip==24.1

pip install -r requirements_nuplan-devkit_fixed.txt
# 以下のエラーは気にしなくて良い
#   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#   cudf 24.4.0 requires protobuf<5,>=3.20, but you have protobuf 5.29.4 which is incompatible.

pip install -r requirements.txt

pip install -e .

# torchの確認
python3 -c "import torch"

# rosの導入
./util_scripts/download_ros-hubmle.sh
```
