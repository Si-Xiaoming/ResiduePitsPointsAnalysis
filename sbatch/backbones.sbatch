#!/bin/bash
#SBATCH --job-name=residue_exp
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=24576
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 注意：删除了 #SBATCH --array，因为集群不支持
# 注意：日志文件名改为 %j (Job ID)，因为没有 %a (Array ID) 了

source activate pointcept
WORK_DIR="/home/shsi/codes/ResiduePitsPointsAnalysis"
cd "$WORK_DIR"
export PYTHONPATH=./

# -----------------------------------------------------------
# 关键修改：获取任务 ID
# 如果没有传入 MY_TASK_ID，默认为 0
TASK_ID=${MY_TASK_ID:-0}
# -----------------------------------------------------------

CONFIG_LIST=(
    "configs/exp_paper/backbone_oacnn.py"
    "configs/exp_paper/backbone_ptv1.py"
    "configs/exp_paper/backbone_ptv2.py"
    "configs/exp_paper/backbone_ptv3.py"
    "configs/exp_paper/backbone_spunet.py"
)

# 使用 TASK_ID 获取配置
CONFIG_PATH=${CONFIG_LIST[$TASK_ID]}

# 检查文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    echo "Current Task ID: $TASK_ID"
    exit 1
fi

CONFIG_NAME=$(basename "${CONFIG_PATH}" .py)
BASE_SAVE_PATH="/home/shsi/outputs/residue/residue-backbone"
SAVE_PATH="${BASE_SAVE_PATH}/${CONFIG_NAME}"

echo "---------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID (Manual): $TASK_ID"
echo "Config: $CONFIG_PATH"
echo "Save Path: $SAVE_PATH"
echo "---------------------------------------"

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((10000 + $RANDOM % 20000))

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    tools/train_torchrun.py \
    --config-file ${CONFIG_PATH} \
    --options save_path=${SAVE_PATH}