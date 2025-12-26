#!/bin/bash
# 启动 Jupyter Lab，无密码模式，根目录为 /workspace
echo "Starting Jupyter Lab..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/workspace