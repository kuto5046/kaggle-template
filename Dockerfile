# pytorch versionに注意
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Define build arguments
ARG DOCKER_UID
ARG DOCKER_USER
ARG DOCKER_PASSWORD

# 時間設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# install basic dependencies
RUN apt-get -y update && apt-get install -y \
    sudo \
    wget \
    cmake \
    vim \
    git \
    tmux \
    zip \
    unzip \
    gcc \
    g++ \
    build-essential \
    ca-certificates \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libsndfile1 \
    libffi-dev \
    libsqlite3-dev \
    libreadline6-dev \
    libbz2-dev \
    libssl-dev \
    libncursesw5-dev \
    libdb-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    libgdbm-dev \
    libmpdec-dev \
    zsh \
    curl


RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
  && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd
USER ${DOCKER_USER}

# Install Sheldon (shell prompt manager)
RUN curl --proto '=https' -fLsS https://rossmacarthur.github.io/install/crate.sh \
| bash -s -- --repo rossmacarthur/sheldon --to ~/.local/bin

# Set environment variables
ENV HOME=/home/${DOCKER_USER}
ENV SHELL=/usr/bin/zsh
WORKDIR ${HOME}

# install dotfiles
RUN git clone https://github.com/kuto5046/dotfiles.git
RUN bash ./dotfiles/.bin/install.sh

# Install mise
RUN curl https://mise.run | sh \
  && echo 'eval "$(mise activate zsh)"' >> ~/.zshrc

# Create and set working directory
RUN mkdir ${HOME}/work/
WORKDIR ${HOME}/work/
