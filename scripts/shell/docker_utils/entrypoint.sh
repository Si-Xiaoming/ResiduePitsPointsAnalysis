#!/bin/bash
set -e

# 1. 确保 SSH 运行目录存在 (有些容器重启会清理 /var/run)
if [ ! -d "/var/run/sshd" ]; then
  mkdir -p /var/run/sshd
fi

# 2. 启动 SSH 服务 (后台运行)
echo "[INFO] Starting SSH Daemon..."
/usr/sbin/sshd

# 3. 打印欢迎信息
echo "=========================================================="
echo "Pointcept Environment Ready"
echo "SSH Port: 22 (User: root, Pass: root)"
echo "Jupyter : run_jupyter (Port 8888)"
echo "TB      : run_tensorboard <log_dir> (Port 6006)"
echo "=========================================================="

# 4. 执行 CMD 命令 (通常是 bash)
exec "$@"