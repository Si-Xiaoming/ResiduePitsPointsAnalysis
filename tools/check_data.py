import os
import math
import shutil  # <--- 新增：用于移动文件和文件夹

def get_dir_size(path):
    """
    计算文件夹的总大小（字节）
    """
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
    """
    格式化大小输出
    """
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def process_and_move_folders(root_path, dest_path, threshold_kb=1024):
    """
    遍历文件夹，并将小于阈值的文件夹移动到 dest_path
    """
    # 1. 基础检查
    if not os.path.exists(root_path):
        print(f"错误：源路径 '{root_path}' 不存在。")
        return
    
    # 2. 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_path):
        print(f"提示：目标文件夹 '{dest_path}' 不存在，正在创建...")
        os.makedirs(dest_path)

    # 防止源目录和目标目录相同
    if os.path.abspath(root_path) == os.path.abspath(dest_path):
        print("错误：源目录和目标目录不能相同！")
        return

    print(f"{'文件夹名称':<40} | {'大小':<10} | {'状态'}")
    print("-" * 80)

    moved_count = 0
    skipped_count = 0
    errors = []

    try:
        items = os.listdir(root_path)
    except PermissionError:
        print(f"没有权限访问: {root_path}")
        return

    for item in items:
        full_source_path = os.path.join(root_path, item)
        
        # 只处理文件夹
        if os.path.isdir(full_source_path):
            # 排除目标文件夹本身（如果目标文件夹在源文件夹里面的话）
            if os.path.abspath(full_source_path) == os.path.abspath(dest_path):
                continue

            size_bytes = get_dir_size(full_source_path)
            size_kb = size_bytes / 1024
            human_readable = format_size(size_bytes)
            
            # 判断是否小于阈值
            if size_kb < threshold_kb:
                # 准备移动
                full_dest_path = os.path.join(dest_path, item)
                
                # 检查目标位置是否已存在同名文件夹
                if os.path.exists(full_dest_path):
                    print(f"{item:<40} | {human_readable:<10} | [跳过] 目标已存在同名文件夹")
                    skipped_count += 1
                else:
                    try:
                        # --- 执行移动操作 ---
                        shutil.move(full_source_path, full_dest_path)
                        print(f"{item:<40} | {human_readable:<10} | [已移动] -> {dest_path}")
                        moved_count += 1
                    except Exception as e:
                        print(f"{item:<40} | {human_readable:<10} | [错误] {str(e)}")
                        errors.append(item)
            else:
                # 大于阈值的文件夹，不做操作，仅显示（可选）
                # print(f"{item:<40} | {human_readable:<10} | [保留] 大于阈值")
                pass

    # --- 最终统计 ---
    print("-" * 80)
    print(f"【处理完成】")
    print(f"源目录: {root_path}")
    print(f"目标目录: {dest_path}")
    print(f"阈值标准: {threshold_kb/1024:.2f} MB")
    print(f"统计结果: 成功移动 {moved_count} 个文件夹，跳过 {skipped_count} 个重复文件夹。")
    if errors:
        print(f"发生错误: {len(errors)} 个 (可能是权限不足)")

if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 1. 原始数据所在的文件夹
    source_dir = "/home/shsi/datasets/Point_Cloud/navarra-10/processed/train"
    
    # 2. 小文件夹要移动到的地方 (如果不存在会自动创建)
    destination_dir = "/home/shsi/datasets/Point_Cloud/avarra_trashed/10_train"
    
    # 3. 阈值: 1 MB = 1024 KB
    size_threshold = 500 
    
    process_and_move_folders(source_dir, destination_dir, size_threshold)