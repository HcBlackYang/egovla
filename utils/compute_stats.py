# # utils/compute_stats.py
# import h5py
# import numpy as np
# import argparse
# import json
# import os
# from tqdm import tqdm

# def compute_stats(args):
#     print(f"Computing stats for {args.data_root} ...")
    
#     all_qpos = []
    
#     with h5py.File(args.data_root, 'r') as f:
#         demos = list(f['data'].keys())
#         for demo_key in tqdm(demos):
#             demo_grp = f['data'][demo_key]
#             # 读取所有关节角度
#             qpos = demo_grp['obs']['robot0_joint_pos'][:] 
            
#             # 处理夹爪拼接逻辑 (7维 -> 8维)
#             if qpos.shape[1] == 7:
#                 # 尝试读取 gripper
#                 if 'robot0_gripper_qpos' in demo_grp['obs']:
#                     gripper = demo_grp['obs']['robot0_gripper_qpos'][:]
#                 elif 'gripper_states' in demo_grp['obs']:
#                      gripper = demo_grp['obs']['gripper_states'][:]
#                 else:
#                     # 默认全 0
#                     gripper = np.zeros((qpos.shape[0], 1))
                
#                 # 拼接成 8 维
#                 if gripper.ndim == 1: gripper = gripper[:, None]
#                 qpos = np.concatenate([qpos, gripper], axis=1)
                
#             all_qpos.append(qpos)
            
#     # 拼接所有数据 [Total_Frames, 8]
#     all_qpos = np.concatenate(all_qpos, axis=0) 
    
#     # 1. 计算原始统计量
#     mean = np.mean(all_qpos, axis=0)
#     std = np.std(all_qpos, axis=0)
    
#     # =========================================================
#     # 🚨 [关键修复] 强制覆盖夹爪统计量 (归一化陷阱修复)
#     # =========================================================
#     # 目的：忽略数据的统计分布，强制将夹爪的物理范围映射到 [-1, 1]
#     # 这样模型不需要预测极端的数值 (如 -4.75)，只需要预测 -1 或 1
    
#     gripper_idx = 7
#     # 获取夹爪的物理极值 (例如: 0.0 ~ 0.08 或 -1 ~ 1)
#     gripper_data = all_qpos[:, gripper_idx]
#     g_min = np.min(gripper_data)
#     g_max = np.max(gripper_data)
    
#     print(f"📊 检测到夹爪物理范围: Min={g_min:.4f}, Max={g_max:.4f}")
    
#     # 计算新的映射参数
#     # 公式: normalized = (x - mean) / std
#     # 我们希望: x=g_max -> 1, x=g_min -> -1
#     # 解方程得:
#     new_mean = (g_max + g_min) / 2.0
#     new_std  = (g_max - g_min) / 2.0
    
#     # 防止除以0 (如果夹爪全程不动)
#     if new_std < 1e-6: 
#         new_std = 1.0
#         print("⚠️ 警告：夹爪数据似乎没有变化，Std设为1.0")

#     # 覆盖
#     mean[gripper_idx] = new_mean
#     std[gripper_idx]  = new_std
    
#     print(f"✅ 已修正夹爪统计量 -> Mean: {mean[gripper_idx]:.4f}, Std: {std[gripper_idx]:.4f}")
#     # =========================================================
    
#     stats = {
#         "action_mean": mean.tolist(),
#         "action_std": std.tolist()
#     }
    
#     # 保存
#     with open(args.save_path, 'w') as f:
#         json.dump(stats, f, indent=4)
        
#     print(f"✅ Stats saved to {args.save_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, required=True, help="Path to HDF5 file")
#     parser.add_argument('--save_path', type=str, default='/yanghaochuan/data/115dataset_stats.json')
#     args = parser.parse_args()
#     compute_stats(args)

# # utils/compute_stats.py
# import h5py
# import numpy as np
# import argparse
# import json
# import os
# from tqdm import tqdm

# def compute_stats(args):
#     print(f"Computing stats for {args.data_root} ...")
#     print(f"⚖️  Balancing Strategy: Type A (Weight 1) | Type B (Weight 4)")
    
#     all_qpos = []
#     type_a_count = 0
#     type_b_count = 0
    
#     with h5py.File(args.data_root, 'r') as f:
#         demos = list(f['data'].keys())
        
#         for demo_key in tqdm(demos):
#             demo_grp = f['data'][demo_key]
            
#             # --- 1. 读取原始数据 ---
#             # 读取所有关节角度
#             qpos = demo_grp['obs']['robot0_joint_pos'][:] 
            
#             # 处理夹爪拼接逻辑 (7维 -> 8维)
#             if qpos.shape[1] == 7:
#                 # 尝试读取 gripper
#                 if 'robot0_gripper_qpos' in demo_grp['obs']:
#                     gripper = demo_grp['obs']['robot0_gripper_qpos'][:]
#                 elif 'gripper_states' in demo_grp['obs']:
#                      gripper = demo_grp['obs']['gripper_states'][:]
#                 else:
#                     # 默认全 0
#                     gripper = np.zeros((qpos.shape[0], 1))
                
#                 # 拼接成 8 维
#                 if gripper.ndim == 1: gripper = gripper[:, None]
#                 qpos = np.concatenate([qpos, gripper], axis=1)
            
#             # --- 2. 判定类型并加权 ---
#             # 假设命名规则是 demo_0, demo_1 ...
#             try:
#                 idx = int(demo_key.split('_')[1])
#             except:
#                 idx = 0 # Fallback
            
#             # Type B (High Start) = ID is multiple of 5
#             if idx % 5 == 0:
#                 weight = 4
#                 type_b_count += 1
#             else:
#                 weight = 1
#                 type_a_count += 1
                
#             # --- 3. 加权收集 ---
#             # 将数据重复 weight 次加入列表
#             # 注意：这不会显著增加内存压力，因为只是引用或小规模拷贝 (qpos数据量通常不大)
#             for _ in range(weight):
#                 all_qpos.append(qpos)
            
#     print(f"📊 Original Demos: Type A={type_a_count}, Type B={type_b_count}")
#     print(f"⚖️  Weighted Ratio: {type_a_count * 1} : {type_b_count * 4} (Approx 1:1)")

#     # 拼接所有数据 [Total_Weighted_Frames, 8]
#     all_qpos_concat = np.concatenate(all_qpos, axis=0) 
    
#     # 4. 计算统计量
#     mean = np.mean(all_qpos_concat, axis=0)
#     std = np.std(all_qpos_concat, axis=0)
    
#     # =========================================================
#     # 🚨 [保留] 强制覆盖夹爪统计量 (归一化修复)
#     # =========================================================
#     gripper_idx = 7
#     gripper_data = all_qpos_concat[:, gripper_idx]
#     g_min = np.min(gripper_data)
#     g_max = np.max(gripper_data)
    
#     print(f"📊 Detected Gripper Range: Min={g_min:.4f}, Max={g_max:.4f}")
    
#     # 映射到 [-1, 1]
#     new_mean = (g_max + g_min) / 2.0
#     new_std  = (g_max - g_min) / 2.0
    
#     if new_std < 1e-6: 
#         new_std = 1.0
#         print("⚠️ Warning: Gripper static, Std set to 1.0")

#     mean[gripper_idx] = new_mean
#     std[gripper_idx]  = new_std
    
#     print(f"✅ Corrected Gripper Stats -> Mean: {mean[gripper_idx]:.4f}, Std: {std[gripper_idx]:.4f}")
    
#     # 5. 保存
#     stats = {
#         "action_mean": mean.tolist(),
#         "action_std": std.tolist()
#     }
    
#     with open(args.save_path, 'w') as f:
#         json.dump(stats, f, indent=4)
        
#     print(f"✅ Weighted Stats saved to {args.save_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, required=True, help="Path to HDF5 file")
#     # 建议保存为新文件名，以免覆盖旧的对比
#     parser.add_argument('--save_path', type=str, default='/yanghaochuan/data/121dataset_stats.json')
#     args = parser.parse_args()
#     compute_stats(args)

# import h5py
# import numpy as np
# import argparse
# import json
# import os
# from tqdm import tqdm

# def compute_stats(args):
#     print(f"Computing stats for {args.data_root} ...")
    
#     all_qpos = []
    
#     with h5py.File(args.data_root, 'r') as f:
#         demos = list(f['data'].keys())
        
#         for demo_key in tqdm(demos):
#             demo_grp = f['data'][demo_key]
#             qpos = demo_grp['obs']['robot0_joint_pos'][:] 
            
#             if qpos.shape[1] == 7:
#                 if 'robot0_gripper_qpos' in demo_grp['obs']:
#                     gripper = demo_grp['obs']['robot0_gripper_qpos'][:]
#                 elif 'gripper_states' in demo_grp['obs']:
#                      gripper = demo_grp['obs']['gripper_states'][:]
#                 else:
#                     gripper = np.zeros((qpos.shape[0], 1))
#                 if gripper.ndim == 1: gripper = gripper[:, None]
#                 qpos = np.concatenate([qpos, gripper], axis=1)
            
#             try:
#                 idx = int(demo_key.split('_')[1])
#             except:
#                 idx = 0 
            
#             if idx % 5 == 0: weight = 4
#             else: weight = 1
                
#             for _ in range(weight):
#                 all_qpos.append(qpos)
            
#     all_qpos_concat = np.concatenate(all_qpos, axis=0) 
    
#     mean = np.mean(all_qpos_concat, axis=0)
#     std = np.std(all_qpos_concat, axis=0)
    
#     print("\n" + "="*50)
#     print("🏥 SURGICAL STATS CORRECTION (Final Polish)")
#     print("="*50)
    
#     # =========================================================
#     # 🟢 [最终微调]
#     # =========================================================
#     # J2: 极小值 -> 0.05 (防止除零，保持静默)
#     # J3, J5: 离群值 -> 0.40 (解决瞬移)
#     # J4: 移除特殊处理，让它回落到默认的 0.1 (0.07 -> 0.1) 增加稳健性
    
#     TARGETED_STD = {
#         2: 0.05,  # J2: 保持静默
#         3: 0.40,  # J3: 修复瞬移
#         5: 0.40,  # J5: 修复瞬移
#     }
#     DEFAULT_MIN_STD = 0.1 # 其他关节 (包括 J4) 的健康底线
    
#     for i in range(7):
#         original_std = std[i]
        
#         # 1. 特殊名单 (J2, J3, J5)
#         if i in TARGETED_STD:
#             target = TARGETED_STD[i]
#             if original_std < target:
#                 print(f"   💉 Joint {i} [TARGETED]: Too tight ({original_std:.4f}). BOOSTING to {target:.4f}.")
#                 std[i] = target
#             else:
#                 print(f"   ✅ Joint {i} [TARGETED]: Original ({original_std:.4f}) is sufficient.")
        
#         # 2. 默认名单 (J0, J1, J4, J6)
#         else:
#             if original_std < DEFAULT_MIN_STD:
#                 print(f"   ⚠️ Joint {i} [DEFAULT]: Too tight ({original_std:.4f}). Clamping to {DEFAULT_MIN_STD}.")
#                 std[i] = DEFAULT_MIN_STD
#             else:
#                 print(f"   ✅ Joint {i} [DEFAULT]: Healthy ({original_std:.4f}). Keeping original.")

#     # 夹爪
#     gripper_idx = 7
#     gripper_data = all_qpos_concat[:, gripper_idx]
#     g_min = np.min(gripper_data)
#     g_max = np.max(gripper_data)
#     new_mean = (g_max + g_min) / 2.0
#     new_std  = (g_max - g_min) / 2.0
#     if new_std < 1e-6: new_std = 1.0
#     mean[gripper_idx] = new_mean
#     std[gripper_idx]  = new_std
    
#     stats = {
#         "action_mean": mean.tolist(),
#         "action_std": std.tolist()
#     }
    
#     with open(args.save_path, 'w') as f:
#         json.dump(stats, f, indent=4)
        
#     print(f"✅ FINAL Stats saved to {args.save_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, required=True)
#     parser.add_argument('--save_path', type=str, default='/yanghaochuan/data/124dataset_stats_final.json')
#     args = parser.parse_args()
#     compute_stats(args)


import h5py
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

def compute_stats(args):
    print(f"Computing stats for {args.data_root} ...")
    
    all_qpos = []
    
    # 计数器
    count_type_a = 0
    count_type_b = 0
    count_type_c = 0
    
    with h5py.File(args.data_root, 'r') as f:
        demos = list(f['data'].keys())
        print(f"⚖️  Calculating Weighted Stats (Target 80:80:80)...")

        for demo_key in tqdm(demos):
            demo_grp = f['data'][demo_key]
            qpos = demo_grp['obs']['robot0_joint_pos'][:] 
            
            # 处理夹爪 (7->8 维)
            if qpos.shape[1] == 7:
                if 'robot0_gripper_qpos' in demo_grp['obs']:
                    gripper = demo_grp['obs']['robot0_gripper_qpos'][:]
                elif 'gripper_states' in demo_grp['obs']:
                     gripper = demo_grp['obs']['gripper_states'][:]
                else:
                    gripper = np.zeros((qpos.shape[0], 1))
                if gripper.ndim == 1: gripper = gripper[:, None]
                qpos = np.concatenate([qpos, gripper], axis=1)
            
            # # === 权重分配逻辑 (必须与 DataLoader 一致) ===
            # try:
            #     curr_idx = int(demo_key.split('_')[1])
            # except:
            #     curr_idx = 0 
            
            # weight = 1
            # if curr_idx < 100:
            #     # 旧数据
            #     if curr_idx % 5 == 0:
            #         weight = 4 # Type B
            #         count_type_b += 1
            #     else:
            #         weight = 1 # Type A
            #         count_type_a += 1
            # else:
            #     # 新数据 (Type C)
            #     weight = 4     # Type C
            #     count_type_c += 1
                
            # # 加权收集
            # for _ in range(weight):
            #     all_qpos.append(qpos)
            all_qpos.append(qpos)
            
    print(f"📊 Original Counts -> A: {count_type_a} | B: {count_type_b} | C: {count_type_c}")
    print(f"⚖️  Effective Counts -> A: {count_type_a*1} | B: {count_type_b*4} | C: {count_type_c*2}")

    # 拼接
    all_qpos_concat = np.concatenate(all_qpos, axis=0) 
    
    # 计算统计量
    mean = np.mean(all_qpos_concat, axis=0)
    std = np.std(all_qpos_concat, axis=0)
    
    print("\n" + "="*50)
    print("🏥 SURGICAL STATS CORRECTION (Final Polish)")
    print("="*50)
    
    # # =========================================================
    # # 🟢 [Surgical Correction] 针对特定关节的修复
    # # =========================================================
    # # J2: 保持静默 (0.05)
    # # J3, J5: 修复瞬移 (0.40)
    # # 其他: 默认底线 (0.1)
    
    # TARGETED_STD = {
    #     2: 0.05,  
    #     3: 0.40,  
    #     5: 0.40,  
    # }
    # DEFAULT_MIN_STD = 0.1 
    
    # for i in range(7):
    #     original_std = std[i]
        
    #     # 1. 特殊名单
    #     if i in TARGETED_STD:
    #         target = TARGETED_STD[i]
    #         if original_std < target:
    #             print(f"   💉 Joint {i} [TARGETED]: Too tight ({original_std:.4f}). BOOSTING to {target:.4f}.")
    #             std[i] = target
    #         else:
    #             print(f"   ✅ Joint {i} [TARGETED]: Original ({original_std:.4f}) is sufficient.")
        
    #     # 2. 默认名单
    #     else:
    #         if original_std < DEFAULT_MIN_STD:
    #             print(f"   ⚠️ Joint {i} [DEFAULT]: Too tight ({original_std:.4f}). Clamping to {DEFAULT_MIN_STD}.")
    #             std[i] = DEFAULT_MIN_STD
    #         else:
    #             print(f"   ✅ Joint {i} [DEFAULT]: Healthy ({original_std:.4f}). Keeping original.")

    # 3. 夹爪归一化修正
    # 强制将物理极值映射到 [-1, 1]
    gripper_idx = 7
    gripper_data = all_qpos_concat[:, gripper_idx]
    g_min = np.min(gripper_data)
    g_max = np.max(gripper_data)
    
    new_mean = (g_max + g_min) / 2.0
    new_std  = (g_max - g_min) / 2.0
    
    # 防止完全不动
    if new_std < 1e-6: new_std = 1.0
    
    mean[gripper_idx] = new_mean
    std[gripper_idx]  = new_std
    print(f"   🔧 Gripper (J7): Forced range [{g_min:.2f}, {g_max:.2f}] -> [-1, 1]")
    
    # 保存
    stats = {
        "action_mean": mean.tolist(),
        "action_std": std.tolist()
    }
    
    with open(args.save_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ FINAL Weighted Stats saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='/yanghaochuan/data/23dataset_stats.json')
    args = parser.parse_args()
    compute_stats(args)