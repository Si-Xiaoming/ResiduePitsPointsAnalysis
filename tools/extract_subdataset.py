import os
import shutil
import numpy as np
from tqdm import tqdm

def get_folder_size(folder_path):
    """递归计算文件夹大小（单位：字节）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # 跳过链接文件
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def weighted_sample_copy(src_root, dst_root, sample_ratio=0.2):
    """
    遍历源目录，按文件夹大小作为权重进行采样，并复制到目标目录。
    
    Args:
        src_root (str): 源数据集路径 (例如 processed/train)
        dst_root (str): 目标存放路径
        sample_ratio (float): 采样比例 (0.0 - 1.0)
    """
    
    # 1. 检查路径
    if not os.path.exists(src_root):
        print(f"错误: 源路径 {src_root} 不存在")
        return

    # 创建目标文件夹
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
        print(f"已创建目标目录: {dst_root}")

    # 2. 遍历并计算大小
    subfolders = [f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))]
    folder_sizes = []
    
    print(f"正在扫描并计算 {len(subfolders)} 个文件夹的大小...")
    
    valid_folders = [] # 用于存储大小不为0的文件夹
    valid_sizes = []

    for folder in tqdm(subfolders, desc="计算大小"):
        full_path = os.path.join(src_root, folder)
        size = get_folder_size(full_path)
        
        # 只有大小大于0的文件夹才参与采样，防止除以0错误
        if size > 0:
            valid_folders.append(folder)
            valid_sizes.append(size)

    if not valid_folders:
        print("没有找到有效的非空文件夹。")
        return

    # 3. 计算权重 (Probability)
    # 文件夹越大，权重越高
    valid_sizes = np.array(valid_sizes, dtype=np.float64)
    total_size = np.sum(valid_sizes)
    probabilities = valid_sizes / total_size

    # 4. 执行加权采样
    num_samples = int(len(valid_folders) * sample_ratio)
    # 确保至少抽取1个，且不超过总数
    num_samples = max(1, min(num_samples, len(valid_folders)))

    print(f"\n准备抽取: {num_samples} 个文件夹 (采样比例: {sample_ratio})")
    print(f"采样策略: 基于文件大小加权 (Size-Weighted)")

    # replace=False 表示不可重复抽取
    selected_folders = np.random.choice(
        valid_folders, 
        size=num_samples, 
        replace=False, 
        p=probabilities
    )

    # 5. 复制文件
    print(f"\n开始复制到: {dst_root}")
    success_count = 0
    
    for folder_name in tqdm(selected_folders, desc="复制中"):
        src_path = os.path.join(src_root, folder_name)
        dst_path = os.path.join(dst_root, folder_name)

        try:
            # 如果目标已存在，先删除再复制，或者根据需求跳过
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            
            shutil.copytree(src_path, dst_path)
            success_count += 1
        except Exception as e:
            print(f"\n复制 {folder_name} 失败: {e}")

    print("-" * 50)
    print(f"处理完成! 成功复制 {success_count}/{num_samples} 个文件夹。")

# ================= 配置区域 =================
if __name__ == "__main__":
    # 源数据路径 (根据你之前的描述修改)
    source_dataset_path = "/home/shsi/datasets/Point_Cloud/residue/navarra_pert100_gs01/processed/train"
    
    # 目标存储路径 (新创建的文件夹)
    # /home/shsi/datasets/Point_Cloud/residue/navarra_pert001_gs01
    # /home/shsi/datasets/Point_Cloud/residue/navarra_pert005_gs01
    # /home/shsi/datasets/Point_Cloud/residue/navarra_pert010_gs01
    target_dataset_path = "/home/shsi/datasets/Point_Cloud/residue/navarra_pert010_gs01"
    target_dataset_path = os.path.join(target_dataset_path, "processed", "train")
    if not os.path.exists(target_dataset_path):
        os.makedirs(target_dataset_path)
    
    # 采样比例 (例如 0.2 表示抽取 20%)
    ratio = 0.1  # 1%

    weighted_sample_copy(source_dataset_path, target_dataset_path, ratio)