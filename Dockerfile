# pytorch versionに注意
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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
    libsqlite3-dev \
    libncursesw5-dev \
    libffi-dev \
    libdb-dev \
    libexpat1-dev \
    zlib1g-dev \
    liblzma-dev \
    libgdbm-dev \
    libmpdec-dev \
    zsh \
    xonsh \
    nodejs \
    npm \
    curl

# node js を最新Verにする
RUN npm -y install n -g && \
    n stable && \
    apt purge -y nodejs npm

# neovim v0.9.1
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim.appimage
RUN chmod u+x nvim.appimage
RUN ./nvim.appimage --appimage-extract
# RUN ./squashfs-root/AppRun --version
# RUN sudo mv squashfs-root /
RUN sudo ln -s /squashfs-root/AppRun /usr/bin/nvim

# install just
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# 本当はハードコーディングではなくローカルのidと合わせた方が良い
# https://qiita.com/yohm/items/047b2e68d008ebb0f001
ARG DOCKER_UID=1000
ARG DOCKER_USER="user"
ARG DOCKER_PASSWORD="kuzira"

RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
  && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd
USER ${DOCKER_USER}

# 環境変数設定
ENV HOME=/home/${DOCKER_USER}
ENV PATH ${HOME}/.local/bin:$PATH
ENV PATH /usr/bin:$PATH
ENV PYTHONPATH ${HOME}/work:$PYTHONPATH
ENV SHELL /usr/bin/zsh
ENV PYENV_ROOT ${HOME}/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
WORKDIR ${HOME}

# install dotfiles
RUN git clone https://github.com/kuto5046/dotfiles.git
RUN bash ./dotfiles/.bin/install.sh

# install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN /usr/bin/zsh -c "source ~/.zshrc" && pyenv install 3.10.12 && pyenv global 3.10.12

# make workdir
RUN mkdir ${HOME}/work/
WORKDIR ${HOME}/work/

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry config virtualenvs.in-project true
RUN poetry config virtualenvs.prefer-active-python true
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi
