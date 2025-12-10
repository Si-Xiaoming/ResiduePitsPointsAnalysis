#!/bin/bash
#SBATCH --job-name=pointcept_preprocess
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a40
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err

# 1. 激活环境
source activate pointcept


# 2. 设置工作目录 (请确保此路径是完全正确的绝对路径)
# 根据你提供的上下文，我移除了报错日志中那个看起来错误的 "LTN_SGG_VG"
WORK_DIR="/home/shsi/codes/ResiduePitsPointsAnalysis"

if [ ! -d "$WORK_DIR" ]; then
  echo "错误: 目录 $WORK_DIR 不存在！"
  exit 1
fi

cd "$WORK_DIR"
echo "当前工作目录: $(pwd)"

# 3. 设置环境变量
export PYTHONPATH=./

# 4. 设置参数
RAW_DATA_DIR="/home/shsi/datasets/Point_Cloud/navarra17"
GRID_SIZE=0.1
TILE_SIZE=250
HAS_LABEL=True
TILE_ORIENT=False

# 5. 检查 Python 脚本是否存在
SCRIPT_PATH="pointcept/datasets/preprocessing/navarra/navarra.py"
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "错误: Python 脚本 $SCRIPT_PATH 在当前目录下找不到！"
  exit 1
fi

echo "开始预处理..."
# 注意：反斜杠 \ 后面不要加空格
python "$SCRIPT_PATH" \
    --dataset_root "${RAW_DATA_DIR}" \
    --grid_size ${GRID_SIZE} \
    --tile_size ${TILE_SIZE} \
    --has_label ${HAS_LABEL} \
    --tile_orient ${TILE_ORIENT}

echo "数据预处理完成。"