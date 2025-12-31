#!/bin/bash
#SBATCH --job-name=1-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=24576
#SBATCH --ntasks=32
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

source activate pointcept
WORK_DIR="/home/shsi/codes/ResiduePitsPointsAnalysis"
cd "$WORK_DIR"
export PYTHONPATH=./
echo "当前工作目录: $(pwd)"


CONFIG_PATH=configs/on_sbatch/semseg-std2.py
# semseg-navarra.py     semseg-std2.py
SAVE_PATH=/home/shsi/outputs/residue/residue-seg
# 输出job name
echo "Job Name: $SLURM_JOB_NAME"
SAVE_PATH=${SAVE_PATH}/${SLURM_JOB_NAME}
# 输出每个节点可用的 GPU 数量
echo "每个节点的 GPU 数量: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

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
    