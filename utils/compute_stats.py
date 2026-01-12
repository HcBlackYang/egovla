# utils/compute_stats.py
import h5py
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

def compute_stats(args):
    print(f"Computing stats for {args.data_root} ...")
    
    all_qpos = []
    
    with h5py.File(args.data_root, 'r') as f:
        demos = list(f['data'].keys())
        for demo_key in tqdm(demos):
            demo_grp = f['data'][demo_key]
            # 读取所有关节角度
            qpos = demo_grp['obs']['robot0_joint_pos'][:] 
            
            # 处理夹爪拼接逻辑 (7维 -> 8维)
            if qpos.shape[1] == 7:
                # 尝试读取 gripper
                if 'robot0_gripper_qpos' in demo_grp['obs']:
                    gripper = demo_grp['obs']['robot0_gripper_qpos'][:]
                elif 'gripper_states' in demo_grp['obs']:
                     gripper = demo_grp['obs']['gripper_states'][:]
                else:
                    # 默认全 0
                    gripper = np.zeros((qpos.shape[0], 1))
                
                # 拼接成 8 维
                if gripper.ndim == 1: gripper = gripper[:, None]
                qpos = np.concatenate([qpos, gripper], axis=1)
                
            all_qpos.append(qpos)
            
    # 拼接所有数据 [Total_Frames, 8]
    all_qpos = np.concatenate(all_qpos, axis=0) 
    
    # 1. 计算原始统计量
    mean = np.mean(all_qpos, axis=0)
    std = np.std(all_qpos, axis=0)
    
    # =========================================================
    # 🚨 [关键修复] 强制覆盖夹爪统计量 (归一化陷阱修复)
    # =========================================================
    # 目的：忽略数据的统计分布，强制将夹爪的物理范围映射到 [-1, 1]
    # 这样模型不需要预测极端的数值 (如 -4.75)，只需要预测 -1 或 1
    
    gripper_idx = 7
    # 获取夹爪的物理极值 (例如: 0.0 ~ 0.08 或 -1 ~ 1)
    gripper_data = all_qpos[:, gripper_idx]
    g_min = np.min(gripper_data)
    g_max = np.max(gripper_data)
    
    print(f"📊 检测到夹爪物理范围: Min={g_min:.4f}, Max={g_max:.4f}")
    
    # 计算新的映射参数
    # 公式: normalized = (x - mean) / std
    # 我们希望: x=g_max -> 1, x=g_min -> -1
    # 解方程得:
    new_mean = (g_max + g_min) / 2.0
    new_std  = (g_max - g_min) / 2.0
    
    # 防止除以0 (如果夹爪全程不动)
    if new_std < 1e-6: 
        new_std = 1.0
        print("⚠️ 警告：夹爪数据似乎没有变化，Std设为1.0")

    # 覆盖
    mean[gripper_idx] = new_mean
    std[gripper_idx]  = new_std
    
    print(f"✅ 已修正夹爪统计量 -> Mean: {mean[gripper_idx]:.4f}, Std: {std[gripper_idx]:.4f}")
    # =========================================================
    
    stats = {
        "action_mean": mean.tolist(),
        "action_std": std.tolist()
    }
    
    # 保存
    with open(args.save_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ Stats saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="Path to HDF5 file")
    parser.add_argument('--save_path', type=str, default='/yanghaochuan/data/111dataset_stats.json')
    args = parser.parse_args()
    compute_stats(args)
