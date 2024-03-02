# Use nvidia/cuda:12.3.1-devel-ubuntu22.04 as the base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Update the package lists
RUN apt-get update

# Install git
RUN apt-get install -y git

# Install python3
RUN apt-get install -y python3

# Install pip
RUN apt-get install -y python3-pip

# Install python packages
RUN pip install torch torchvision torchaudio
RUN pip install matplotlib
RUN pip install numba
RUN pip install numpy
RUN pip install transformers
RUN pip install triton
RUN pip install ninja
RUN pip install pandas