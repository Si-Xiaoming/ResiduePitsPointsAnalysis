#!/bin/bash
#SBATCH --job-name=test-laz-inference
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --output=logger/%x-%j.out
#SBATCH --error=logger/%x-%j.err

# 1. 环境加载
source activate pointcept
WORK_DIR="/home/shsi/codes/ResiduePitsPointsAnalysis"
cd "$WORK_DIR"
export PYTHONPATH=./
echo "当前工作目录: $(pwd)"
echo "节点列表: $SLURM_JOB_NODELIST"

# 2. 变量配置 (根据需要修改 CONFIG_PATH)
# 这里的 config 应该是你刚才修改过 dataset 和 tester 的那个配置文件
CONFIG_PATH=configs/test/test_navarra.py

# 输出路径
SAVE_PATH=/home/shsi/outputs/residue/residue-segment-test
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"

# 3. 自动检测 GPU 数量
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 GPU 数量: $GPUS_PER_NODE"

# 4. 运行推理命令
# 注意：tools/test.py 会自动处理多卡并行(DataParallel/DistributedDataParallel)，
# 不需要 torchrun，只需要指定 --num-gpus 即可。

python tools/test.py \
    --config-file ${CONFIG_PATH} \
    --num-gpus ${GPUS_PER_NODE} \
    --options save_path=${SAVE_PATH}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}

echo "推理完成。"