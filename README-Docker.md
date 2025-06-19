# Diffusion Planner - Docker Setup

## Prerequisites
- Docker installed
- At least 8GB available disk space

## Setup Commands

### 1. Clone and Build
```bash
git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git
cd Diffusion-Planner
docker build -t diffusion-planner .
```

### 2. Run Container
```bash
# Basic usage
docker run -it --rm diffusion-planner

# With GPU support
docker run -it --rm --gpus all diffusion-planner

# With data persistence
docker run -it --rm -v $(pwd)/data:/workspace/data diffusion-planner
```

### 3. Download Model Checkpoints (inside container)
```bash
wget -P ./checkpoints https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/args.json
wget -P ./checkpoints https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main/model.pth
```

### 4. Verify Installation
```bash
python3 -c "import diffusion_planner; print('Success')"
```

### 5. Run Simulation
```bash
bash sim_diffusion_planner_runner.sh
```

## Development
```bash
# Mount source code for development
docker run -it --rm -v $(pwd):/workspace/diffusion-planner diffusion-planner bash
```

## Troubleshooting
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Clean up space
docker system prune -a
```