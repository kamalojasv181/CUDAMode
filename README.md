# CUDA MODE

## Introduction
This is [my](https://github.com/kamalojasv181) personal practice repo for the [CUDA MODE](https://github.com/cuda-mode) course.

## Installation
I use a docker container to run the code. You can use the following command to run the container:
```bash
# Build the container
docker build -t cuda-mode .

# Run the container
docker run --gpus all -v /home:/home --ipc=host -it -d --name cuda-mode-container cuda-mode
```
