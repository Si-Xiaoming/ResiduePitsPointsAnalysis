#!/bin/bash
#SBATCH --job-name=extract_subdataset
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
python tools/extract_subdataset.py
