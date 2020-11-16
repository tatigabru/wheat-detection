FROM nvidia/cuda::11.1.0-cudnn8-runtime-ubuntu18.04

## Base packages for ubuntu
RUN apt-get clean && \
    apt-get update -qq && \
    apt-get install -y \
        sudo \
        gosu \
        git \
        wget \
        bzip2 \
        htop \
        nano \
        g++ \
        gcc \
        make \
        build-essential \
        software-properties-common \
        apt-transport-https \
        libhdf5-dev \
        libgl1-mesa-glx \
        openmpi-bin \
        graphviz

## Download and install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

# Install python and upgrade conda and pip version
RUN conda update -n base conda && \ 
    conda install -y python=3.7 && \
    pip install --upgrade pip


ENV IMAGE_PATH="/wdata/pretrained_models/"

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************