#!/bin/bash
#SBATCH --job-name=vis_consistency
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

CONFIG_FILE=configs/on_sbatch/pretrain-sonata-v1m1-0-base-navarra.py
CHECKPOINT=/home/shsi/outputs/on_sbatch/02_std_monitor/model/epoch_10.pth.pth
SAVE_PATH=/home/shsi/outputs/residue/images
DROP_RATE=0.8  # 这里设置丢弃率，建议设置高一点 (0.8 或 0.9) 来凸显效果

python paper_tools/visualize_consistency.py \
  --config-file $CONFIG_FILE \
  --checkpoint $CHECKPOINT \
  --save-path $SAVE_PATH \
  --drop-rate $DROP_RATE