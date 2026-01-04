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
            # 读取所有关节角度 (T, 7)
            qpos = demo_grp['obs']['robot0_joint_pos'][:] 
            
            # 如果您的动作包含夹爪 (8维)，需要在这里处理
            # 假设 dataset_loader 里处理的是 8 维，这里也要由 7 维拼成 8 维
            # 通常 HDF5 里 gripper 可能是单独的 key，或者已经拼好了
            # 这里假设 robot0_joint_pos 只有 7 维，我们需要去读 gripper
            
            if qpos.shape[1] == 7:
                # 尝试读取 gripper
                if 'robot0_gripper_qpos' in demo_grp['obs']:
                    gripper = demo_grp['obs']['robot0_gripper_qpos'][:]
                elif 'gripper_states' in demo_grp['obs']:
                     gripper = demo_grp['obs']['gripper_states'][:]
                else:
                    # 默认全 0 (open) 或 全 1
                    gripper = np.zeros((qpos.shape[0], 1))
                
                # 拼接成 8 维
                if gripper.ndim == 1: gripper = gripper[:, None]
                qpos = np.concatenate([qpos, gripper], axis=1)
                
            all_qpos.append(qpos)
            
    # 拼接所有数据
    all_qpos = np.concatenate(all_qpos, axis=0) # [Total_Frames, 8]
    
    # 计算统计量
    mean = np.mean(all_qpos, axis=0).tolist()
    std = np.std(all_qpos, axis=0).tolist()
    
    stats = {
        "action_mean": mean,
        "action_std": std
    }
    
    # 保存
    with open(args.save_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ Stats saved to {args.save_path}")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="Path to HDF5 file")
    parser.add_argument('--save_path', type=str, default='/yanghaochuan/data/13dataset_stats.json')
    args = parser.parse_args()
    compute_stats(args)