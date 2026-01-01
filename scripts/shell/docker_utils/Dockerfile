# ========================================================================================
# 1. 基础镜像定义
# ========================================================================================
ARG TORCH_VERSION=2.5.0
ARG CUDA_VERSION=12.4
ARG CUDNN_VERSION=9
ARG CUDA_VERSION_NO_DOT=124

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel
ARG TORCH_VERSION
ARG CUDA_VERSION_NO_DOT
# ========================================================================================
# 2. 环境变量设置
# ========================================================================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
ENV DEBIAN_FRONTEND=noninteractive

# ========================================================================================
# 3. 安装系统级依赖
# ========================================================================================
RUN apt-get update && apt-get -y install --no-install-recommends \
      git wget tmux vim zsh build-essential cmake ninja-build \
      libopenblas-dev libsparsehash-dev libgl1-mesa-glx \
      libgdal-dev gdal-bin python3-gdal \
      openssh-server curl iputils-ping nano htop net-tools \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# ========================================================================================
# 4. 配置 SSH 服务 (构建阶段的基础配置)
# ========================================================================================
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    # 修复 Docker 中 SSH 登录可能出现的 pam 限制问题
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# ========================================================================================
# 5. 安装 Python/Conda 依赖
# ========================================================================================
# 安装 Conda/Mamba
RUN conda install -c conda-forge mamba -y

# 安装基础科学计算包和 Jupyter
RUN mamba install -y \
    h5py pyyaml tensorboard tensorboardx wandb yapf addict einops scipy \
    plyfile termcolor matplotlib black open3d \
    pdal python-pdal=3.2.3 \
    jupyterlab  # 显式安装 jupyterlab

# 安装 Pip 包
RUN pip install --upgrade pip
RUN pip install timm numpy==1.26.4
# PyG 安装

RUN pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT}.html

# Spconv
RUN pip install spconv-cu124
# 其他工具
RUN pip install ftfy regex tqdm omegaconf argparse scikit-learn SharedArray

# ========================================================================================
# 6. 编译安装重型 CUDA 依赖 (FlashAttn, OCNN, Swin3D)
# ========================================================================================
RUN pip install git+https://github.com/octree-nn/ocnn-pytorch.git
RUN pip install git+https://github.com/openai/CLIP.git
ENV MAX_JOBS=2
RUN pip install -U git+https://github.com/microsoft/Swin3D.git --no-build-isolation -v
# Flash Attention 编译较慢，如有必要可注释掉
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation

# ========================================================================================
# 7. 预编译 Pointcept 核心算子
# ========================================================================================
# 我们将代码 clone 到 /opt/Pointcept，并编译算子到系统环境
# 这样即使您挂载了自己的代码到 /workspace，只要 import pointops 依然可用
WORKDIR /opt
RUN git clone https://github.com/Pointcept/Pointcept.git
WORKDIR /opt/Pointcept/libs/pointops
RUN python setup.py install
WORKDIR /opt/Pointcept/libs/pointgroup_ops
RUN python setup.py install

# ========================================================================================
# 8. 启动脚本配置
# ========================================================================================
COPY run_jupyter.sh /usr/local/bin/run_jupyter
COPY run_tensorboard.sh /usr/local/bin/run_tensorboard
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/run_jupyter && \
    chmod +x /usr/local/bin/run_tensorboard && \
    chmod +x /usr/local/bin/entrypoint.sh

# 暴露端口: 22(SSH), 8888(Jupyter), 6006(Tensorboard)
EXPOSE 22 8888 6006

# 工作目录设置为 workspace，等待您挂载代码
WORKDIR /workspace

# 关键：使用 entrypoint 脚本启动 SSH
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]