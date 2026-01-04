import torch
import numpy as np
from utils.dataset_loader import RobotDataset
import matplotlib.pyplot as plt

# 指向你的数据路径
DATA_PATH = '/yanghaochuan/data/12pick_up_the_orange_ball.hdf5'

def check_gripper():
    print("正在检查数据集夹爪数据...")
    ds = RobotDataset(hdf5_path=DATA_PATH, window_size=16, pred_horizon=64)
    
    gripper_variance_found = False
    
    # 随机抽查 10 个样本
    indices = np.random.choice(len(ds), 10)
    
    for i in indices:
        sample = ds[i]
        actions = sample['action_target'] # [64, 8]
        
        # 取第 8 维 (索引 7)
        gripper_vals = actions[:, 7].numpy()
        
        # 反归一化看看真实值 (假设 0是开, 1是关)
        mean_g = ds.action_mean[7].item()
        std_g = ds.action_std[7].item()
        raw_gripper = gripper_vals * std_g + mean_g
        
        print(f"Sample {i} Gripper Raw Range: [{raw_gripper.min():.4f}, {raw_gripper.max():.4f}]")
        
        if raw_gripper.max() - raw_gripper.min() > 0.1:
            gripper_variance_found = True
            print("✅ 发现夹爪动作变化！")
            
    if not gripper_variance_found:
        print("\n❌ 警告：抽样的样本中夹爪似乎没有动过！")
        print("这确认了 dataset_loader.py 没有正确读取夹爪数据。")
        print("请按上面的建议修改 dataset_loader.py 并重新训练。")
    else:
        print("\n✅ 数据集读取正常。问题可能出在推理代码。")

if __name__ == "__main__":
    check_gripper()