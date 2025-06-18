# Use nvidia/cuda:12.3.1-devel-ubuntu22.04 as the base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Update the package lists and install system dependencies in a single layer
RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

# Install python packages in a single layer to reduce image size
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    matplotlib \
    numba \
    numpy \
    transformers \
    triton \
    ninja \
    pandas