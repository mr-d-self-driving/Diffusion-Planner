FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base packages
RUN pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# Clone and install nuplan-devkit
RUN git clone https://github.com/motional/nuplan-devkit.git && \
    cd nuplan-devkit && \
    pip install "pip<24.1" && \
    pip install -e . && \
    pip install -r requirements.txt

# Copy the diffusion planner code
COPY . /workspace/diffusion-planner/

# Install diffusion planner requirements
RUN cd /workspace/diffusion-planner && \
    pip install -r ./diffusion_planner/requirements.txt && \
    pip install -e ./diffusion_planner

# Create checkpoints directory for model weights
RUN mkdir -p /workspace/diffusion-planner/checkpoints

# Set the working directory to diffusion-planner
WORKDIR /workspace/diffusion-planner

CMD ["bash"]
