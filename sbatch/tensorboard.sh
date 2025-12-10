# 激活您的 conda 环境
source activate pointcept

# 设置日志目录，替换为您的实际路径
LOG_DIR="/home/shsi/outputs/on_sbatch/pretrain-sonata-v1m1-0-outdoor-navarra"
REMOTE_PORT=6006 

# 启动 TensorBoard 服务
# --bind_all 允许从集群外部访问
tensorboard --logdir="$LOG_DIR" --port=$REMOTE_PORT --bind_all


tensorboard --logdir="/home/shsi/outputs/on_sbatch/pretrain-sonata-v1m1-0-outdoor-navarra" --port=1352 --bind_all