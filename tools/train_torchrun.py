# tools/train_torchrun.py

import os
import torch
import torch.distributed as dist
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.utils import comm

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def init_dist_environment():
    """
    初始化分布式环境，模拟 pointcept.engines.launch 中的逻辑，
    但是基于 torchrun 设置的环境变量。
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("未检测到分布式环境变量，尝试以非分布式模式运行...")
        return
    
    # 1. 从环境变量获取 RANK, WORLD_SIZE, LOCAL_RANK
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 注意：torchrun 默认不提供 num_gpus_per_machine (LOCAL_WORLD_SIZE)，但通常可以获取
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))

    # 2. 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # 3. 初始化进程组
    # torchrun 会自动处理 MASTER_ADDR 和 MASTER_PORT
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

    # 4. 重建 Pointcept 所需的本地进程组 (Local Process Group)
    # Pointcept 的 comm.py 依赖 _LOCAL_PROCESS_GROUP 来获取本地 rank
    num_machines = world_size // local_world_size
    machine_rank = rank // local_world_size
    
    for i in range(num_machines):
        ranks_on_i = list(range(i * local_world_size, (i + 1) * local_world_size))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg
            
    # 同步确保初始化完成
    comm.synchronize()

def main():
    # 使用 Pointcept 默认的参数解析器
    args = default_argument_parser().parse_args()
    
    # 初始化分布式环境
    init_dist_environment()
    
    # 解析配置
    cfg = default_config_parser(args.config_file, args.options)

    # 直接运行 Worker，不再使用 launch()
    main_worker(cfg)

if __name__ == "__main__":
    main()