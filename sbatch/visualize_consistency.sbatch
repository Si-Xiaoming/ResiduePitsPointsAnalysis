#!/bin/bash
#SBATCH --job-name=visualize_consistency
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=24576
#SBATCH --ntasks=32
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source activate pointcept
WORK_DIR="/home/shsi/codes/ResiduePitsPointsAnalysis"
cd "$WORK_DIR"
export PYTHONPATH=./
echo "当前工作目录: $(pwd)"
# vis_consist.py  
CONFIG_FILE=configs/on_sbatch/vis_consist.py
# 指定预训练模型的路径
## /home/shsi/outputs/on_sbatch/02_std_monitor/model/epoch_10.pth

## /home/shsi/outputs/on_sbatch/monitor_gs02/model/epoch_10.pth
## /home/shsi/outputs/on_sbatch/0-0_density/model/epoch_10.pth
## /home/shsi/outputs/residue/residue-pretrain/1-3-pre2/model/epoch_10.pth
CHECKPOINT=/home/shsi/outputs/residue/residue-pretrain/1-3-pre2/model/epoch_10.pth
SAVE_PATH=/home/shsi/outputs/residue/images/02_pre2density_unused
DROP_RATE=0.8  # 这里设置丢弃率，建议设置高一点 (0.8 或 0.9) 来凸显效果


# visualize_consistency.py    vis_tsne.py
python paper_tools/visualize_consistency.py \
  --config-file $CONFIG_FILE \
  --checkpoint $CHECKPOINT \
  --save-path $SAVE_PATH \
  --drop-rate $DROP_RATE