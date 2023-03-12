FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ARG IMAGE_CREATOR
RUN echo $IMAGE_CREATOR > /etc/image_creator
RUN date > /etc/image_date

USER root

##### Install required tools #####
RUN apt update \
	&& echo exit 0 > /usr/sbin/policy-rc.d \
	&& apt-get install -y --no-install-recommends \
					git \
					wget \
					xterm \
					gedit \
					unzip \
					curl \
					bzip2 \
					openjdk-8-jre \
					vim \
					mc \
					eog \
					tmux \
					xclip \
					ssh \
					libopenmpi-dev \
					openssh-server \
 	                net-tools \
					sudo \
					ca-certificates \
					firefox \
	&& DEBIAN_FRONTEND='noninteractive' apt-get -y install tzdata \				
	&& rm -rf /var/lib/apt/lists/*
###################################

# Miniconda3
ARG miniconda3_download_url=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget $miniconda3_download_url --no-check-certificate -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/anaconda3 \
	&& echo ". /opt/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
	&& rm ~/miniconda.sh
RUN /bin/bash -c "source ~/.bashrc"
ENV PATH="opt/anaconda3/bin:$PATH"

# Setup anaconda (1):
# Create anaconda environment:
RUN /bin/bash -c "source activate base && \
	conda create -n blockcopy python=3.9 -y"

# Setup anaconda (2):
# Install pytorch & torchvision with pip:
RUN /bin/bash -c "source activate blockcopy && \
	pip install --upgrade pip && \
	pip install --trusted-host download.pytorch.org https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp39-cp39-linux_x86_64.whl && \
	pip install --trusted-host download.pytorch.org https://download.pytorch.org/whl/cu111/torchvision-0.10.1%2Bcu111-cp39-cp39-linux_x86_64.whl"	

# Setup anaconda (3):
# Other conda installations
RUN /bin/bash -c "source activate blockcopy && \
	conda install ipython pillow numpy opencv future tqdm pyyaml scipy pyyaml tensorboard joblib pandas scikit-image"

# Setup anaconda (4)
# Other pip installations
RUN /bin/bash -c "source activate blockcopy && \
	pip install opencv-contrib-python torchinfo thop torchmetrics tqdm pydevd-pycharm==202.8194.22 matplotlib plotly wandb cmapy ptflops==0.6.6 cupy-cuda111==9.5.0 chardet mmdet==0.6.0 cython pytest-runner six terminaltables pycocotools mmcv==0.4.3"