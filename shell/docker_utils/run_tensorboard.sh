#!/bin/bash
# 默认读取 /workspace/exp，也可以传参数指定目录
LOG_DIR=${1:-/workspace/exp}
mkdir -p $LOG_DIR
echo "Starting Tensorboard on $LOG_DIR..."
tensorboard --logdir "$LOG_DIR" --port 6006 --bind_all