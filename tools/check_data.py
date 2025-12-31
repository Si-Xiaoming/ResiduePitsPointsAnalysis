import os
import math
import shutil
import operator

# --- 工具函数保持不变 ---
def get_dir_size(path):
    """计算文件夹的总大小（字节）"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass
    except PermissionError:
        return 0
    return total_size

def format_size(size_bytes):
    """格式化大小输出 (B, KB, MB...)"""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# --- 新增：核心处理逻辑 ---

def smart_manage_folders(root_path, dest_path, threshold_mb, condition='<', dry_run=True):
    """
    通用文件夹管理函数
    :param root_path: 源目录
    :param dest_path: 目标目录
    :param threshold_mb: 阈值大小 (单位 MB)
    :param condition: 比较条件字符串 ('>', '>=', '<', '<=')
    :param dry_run: True=只看若不移动 (模拟), False=实际移动
    """
    
    # 1. 定义操作符映射
    ops = {
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le
    }
    
    if condition not in ops:
        print(f"错误：不支持的比较条件 '{condition}'。请使用 >, >=, <, <=")
        return

    op_func = ops[condition]
    threshold_bytes = threshold_mb * 1024 * 1024  # 转换为字节以便比较

    # 2. 基础检查
    if not os.path.exists(root_path):
        print(f"错误：源路径 '{root_path}' 不存在。")
        return

    # 如果是实际移动模式，且目标不存在，则创建
    if not dry_run and not os.path.exists(dest_path):
        print(f"提示：目标文件夹 '{dest_path}' 不存在，正在创建...")
        os.makedirs(dest_path)

    # 打印表头
    mode_str = "【模拟预览模式】" if dry_run else "【执行移动模式】"
    print(f"\n当前运行模式: {mode_str}")
    print(f"筛选条件: 大小 {condition} {threshold_mb} MB")
    print("-" * 100)
    print(f"{'文件夹名称':<40} | {'大小':<12} | {'判定结果'}")
    print("-" * 100)

    count_match = 0
    count_moved = 0
    
    try:
        items = os.listdir(root_path)
    except PermissionError:
        print(f"没有权限访问: {root_path}")
        return

    for item in items:
        full_source_path = os.path.join(root_path, item)
        
        # 排除自身和非文件夹
        if not os.path.isdir(full_source_path):
            continue
        if os.path.abspath(full_source_path) == os.path.abspath(dest_path):
            continue

        # 获取大小
        size_bytes = get_dir_size(full_source_path)
        human_readable = format_size(size_bytes)

        # --- 核心判断 ---
        is_match = op_func(size_bytes, threshold_bytes)

        if is_match:
            count_match += 1
            if dry_run:
                # 只是查看
                print(f"{item:<40} | {human_readable:<12} | [符合条件] (待移动)")
            else:
                # 实际移动
                full_dest_path = os.path.join(dest_path, item)
                if os.path.exists(full_dest_path):
                    print(f"{item:<40} | {human_readable:<12} | [跳过] 目标已存在")
                else:
                    try:
                        shutil.move(full_source_path, full_dest_path)
                        print(f"{item:<40} | {human_readable:<12} | [已移动] -> 目标目录")
                        count_moved += 1
                    except Exception as e:
                        print(f"{item:<40} | {human_readable:<12} | [错误] {str(e)}")
        else:
            # 不符合条件的（可选：如果不想看太多日志，可以注释掉下面这行）
            # print(f"{item:<40} | {human_readable:<12} | [忽略] 不满足条件")
            pass

    print("-" * 100)
    if dry_run:
        print(f"统计: 发现 {count_match} 个符合条件的文件夹。")
        print("提示: 当前为预览模式，未移动任何文件。将配置中 dry_run 改为 False 以执行操作。")
    else:
        print(f"统计: 符合条件 {count_match} 个，成功移动 {count_moved} 个。")


if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 源目录
    source_dir = "/home/shsi/datasets/Point_Cloud/navarra-01/processed/val"
    
    # 2. 目标目录 (移动目的地)
    destination_dir = "/home/shsi/datasets/Point_Cloud/navarra-01/processed/val_trash"
    
    # 3. 阈值大小 (单位 MB)
    target_size_mb = 50  # 例如：1 MB
    
    # 4. 比较条件
    # '>'  : 找出大于该大小的文件夹
    # '<'  : 找出小于该大小的文件夹
    condition_operator = '>' 
    
    # 5. 运行模式 (关键设置!)
    # True  : 只查看，不移动 (安全模式，先运行这个看看结果)
    # False : 实际执行移动操作
    is_dry_run = False  
    
    # ===========================================

    smart_manage_folders(
        root_path=source_dir, 
        dest_path=destination_dir, 
        threshold_mb=target_size_mb, 
        condition=condition_operator, 
        dry_run=is_dry_run
    )