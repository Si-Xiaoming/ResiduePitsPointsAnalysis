
CUDA_VISIBLE_DEVICES=0  # which gpu
NUM_GPU=1
CONFIG_PATH=configs/navarra/semseg-sonata-v1m2_std-0a-navarra-lin-outdoor-grid_size10.py
SAVE_PATH=/datasets/exp/semseg-sonata-v1m2_std-0a-navarra-lin-outdoor-grid_size10
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

export PYTHONPATH=./   # Project path
# python -c "import torch; print(torch.cuda.is_available());"
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
