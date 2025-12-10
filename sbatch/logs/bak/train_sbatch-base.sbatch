#!/bin/bash
#SBATCH --job-name=base-navarra
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=24576
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source activate pointcept
WORK_DIR="/home/shsi/codes/ResiduePitsPointsAnalysis"
cd "$WORK_DIR"
export PYTHONPATH=./
echo "当前工作目录: $(pwd)"


CONFIG_PATH=configs/on_sbatch/pretrain-sonata-v1m1-0-base-navarra.py
SAVE_PATH=/home/shsi/outputs/on_sbatch/pretrain-sonata-v1m1-0-base-navarra_bs4-lowstudent_drop_rate_lr0002


# 获取节点和 GPU 信息
GPUS_PER_NODE=4 # 每个节点的 GPU 数量
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + $RANDOM % 20000))

# ---------------- 运行命令 ----------------
# 使用 srun 启动 torchrun (如果是多节点) 或者直接运行 (如果是单节点)
# 下面是通用的单/多节点写法：

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    tools/train_torchrun.py \
    --config-file ${CONFIG_PATH} \
    --options save_path=${SAVE_PATH}
    # nnodes 是节点数，nproc_per_node 是每个节点的 GPU 数量
    # rdzv_id 是 rendezvous ID，通常使用 SLURM_JOB_ID
    # rdzv_backend 是 rendezvous 后端，c10d 是 PyTorch 的默认后端
    # rdzv_endpoint 是 rendezvous 端点，格式为 "主机名:端口号"
    