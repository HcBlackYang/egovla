# # # import h5py
# # # import numpy as np
# # # import argparse

# # # def inspect_hdf5(file_path):
# # #     print(f"正在检查文件: {file_path} ...\n")
    
# # #     try:
# # #         with h5py.File(file_path, 'r') as f:
# # #             data_grp = f['data']
# # #             print(f"总共包含演示条数: {len(data_grp)}")
            
# # #             # 随机抽查前 3 个 demo
# # #             demo_keys = list(data_grp.keys())[:10]
            
# # #             for key in demo_keys:
# # #                 print(f"\n=== 检查 {key} ===")
# # #                 demo = data_grp[key]
                
# # #                 # 1. 检查长度 (验证降采样)
# # #                 qpos = demo['obs']['robot0_joint_pos'][:]
# # #                 img = demo['obs']['agentview_image'][:]
# # #                 actions = demo['actions'][:]
                
# # #                 print(f"数据长度 (Frames): {len(qpos)}")
# # #                 if len(qpos) < 500:
# # #                     print("✅ 长度符合预期 (原始约100帧 -> 降采样后约20帧)")
# # #                 else:
# # #                     print("❌ 长度过长，可能降采样未生效！")

# # #                 # 2. 检查起步运动 (验证静止切除)
# # #                 # 打印前 3 帧的关节角度变化
# # #                 print("\n前 5 帧关节角度 (Joint 0-3):")
# # #                 for i in range(min(5, len(qpos))):
# # #                     print(f"Frame {i}: {qpos[i][:4]}")
                
# # #                 # 计算第0帧和第1帧的平均变化量
# # #                 diff = np.mean(np.abs(qpos[1] - qpos[0]))
# # #                 print(f"\nFrame 0 -> 1 平均变化量: {diff:.6f}")
                
# # #                 if diff > 0.0005: # 0.0005 rad 约等于 0.03度，降采样后变化量应该很大
# # #                     print("✅ 起步即有动作 (静止帧已切除)")
# # #                 else:
# # #                     print("⚠️ 起步变化极小，可能仍包含静止帧")

# # #                 # 3. 检查教师特征 (验证双视角教师)
# # #                 if 'teacher_siglip' in demo and 'teacher_exo' in demo:
# # #                     siglip_shape = demo['teacher_siglip'].shape
# # #                     exo_shape = demo['teacher_exo'].shape
# # #                     print(f"\n✅ 教师特征存在:")
# # #                     print(f"   SigLIP (Global): {siglip_shape}")
# # #                     print(f"   Exo (Wrist):     {exo_shape}")
                    
# # #                     # 检查 Exo 是否全是 0 (验证是否有手腕视频)
# # #                     if np.all(demo['teacher_exo'][:] == 0):
# # #                         print("⚠️ 警告: Exo 特征全为 0 (可能缺少 wrist_image.mp4)")
# # #                     else:
# # #                         print("✅ Exo 特征正常 (非全0)")
# # #                 else:
# # #                     print("❌ 缺少教师特征数据！")

# # #     except Exception as e:
# # #         print(f"无法读取文件: {e}")

# # # if __name__ == "__main__":
# # #     # 修改这里为你生成的 HDF5 路径
# # #     file_path = "/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5" 
# # #     inspect_hdf5(file_path)

# # import h5py
# # import numpy as np
# # import cv2
# # import os

# # def diagnose(file_path):
# #     print(f"🏥 正在诊断 HDF5 文件: {file_path} ...\n")
    
# #     if not os.path.exists(file_path):
# #         print("❌ 文件不存在！")
# #         return

# #     try:
# #         with h5py.File(file_path, 'r') as f:
# #             data = f['data']
# #             print(f"📊 总数据量: {len(data)} 条 Episodes")
            
# #             # === 1. 随机抽查 3 条数据 ===
# #             sample_keys = list(data.keys())[:3]
            
# #             for key in sample_keys:
# #                 print(f"\n--- 检查 {key} ---")
# #                 demo = data[key]
                
# #                 # 读取关键数据
# #                 qpos = demo['obs']['robot0_joint_pos'][:]
# #                 actions = demo['actions'][:]
# #                 imgs = demo['obs']['agentview_image'][:]
                
# #                 # --- A. 长度检查 ---
# #                 T = len(actions)
# #                 print(f"1. 时间步长 (Frames): {T}")
# #                 # 30Hz 下，10秒应该是 300帧左右。如果小于 100 或大于 600 都不对劲
# #                 if 150 <= T <= 450:
# #                     print(f"   ✅ 长度合理 (约 {T/30:.1f} 秒)")
# #                 else:
# #                     print(f"   ⚠️ 长度异常！可能过短或过长")

# #                 # --- B. 维度检查 (最关键!) ---
# #                 print(f"2. Action 维度: {actions.shape}")
# #                 if actions.shape[1] == 8:
# #                     print("   ✅ 维度正确 (7关节 + 1夹爪)")
# #                 else:
# #                     print(f"   ❌ 维度错误！期望 (T, 8), 实际 {actions.shape} (夹爪丢了？)")

# #                 # --- C. 夹爪数值检查 ---
# #                 gripper_data = actions[:, 7] # 第8列
# #                 g_min, g_max = gripper_data.min(), gripper_data.max()
# #                 g_diff = g_max - g_min
# #                 print(f"3. 夹爪活动范围: {g_min:.4f} ~ {g_max:.4f} (Diff: {g_diff:.4f})")
# #                 if g_diff > 0.0001:
# #                     print("   ✅ 夹爪有动作 (数据正常)")
# #                 else:
# #                     print("   ⚠️ 警告：夹爪似乎全程没动 (或全是0)")

# #                 # --- D. 图像与特征检查 ---
# #                 print(f"4. 图像形状: {imgs.shape}")
# #                 if 'teacher_siglip' in demo:
# #                     feat_shape = demo['teacher_siglip'].shape
# #                     print(f"   ✅ Teacher Feature: {feat_shape} (SigLIP)")
# #                 else:
# #                     print("   ❌ 缺少 Teacher Feature")

# #             # === 2. 导出视频 (视觉验证) ===
# #             print(f"\n🎥 正在将 {sample_keys[0]} 还原为视频以供目测...")
# #             save_video_from_hdf5(data[sample_keys[0]], "debug_check_video.mp4")

# #     except Exception as e:
# #         print(f"❌ 读取失败: {e}")

# # def save_video_from_hdf5(group, save_path):
# #     images = group['obs']['agentview_image'][:] # (T, 224, 224, 3)
    
# #     # 初始化视频写入
# #     h, w = images.shape[1], images.shape[2]
# #     out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
# #     for i in range(len(images)):
# #         # HDF5里通常是 RGB，OpenCV 需要 BGR
# #         img_bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
# #         out.write(img_bgr)
    
# #     out.release()
# #     print(f"✅ 视频已保存至: {os.path.abspath(save_path)}")
# #     print("   -> 请下载此视频并在本地播放，检查动作是否丝滑、有无卡顿。")

# # if __name__ == "__main__":
# #     # 修改为你的文件路径
# #     file_path = "/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5"
# #     diagnose(file_path)

# # utils/compute_stats.py
# import h5py
# import numpy as np
# import argparse
# import json
# import os

# def compute_stats(data_root, output_path):
#     print(f"Reading data from {data_root}...")
#     all_qpos = []
    
#     with h5py.File(data_root, 'r') as f:
#         demos = list(f['data'].keys())
#         for key in demos:
#             # 读取 actions 或 robot0_joint_pos (8维: 7关节+1夹爪)
#             qpos = f['data'][key]['actions'][:]
#             all_qpos.append(qpos)
    
#     # 拼接所有数据 [N, 8]
#     all_data = np.concatenate(all_qpos, axis=0)
    
#     # 计算统计量
#     mean = np.mean(all_data, axis=0).tolist()
#     std = np.std(all_data, axis=0).tolist()
    
#     # 防止 std 为 0 (比如夹爪一直没动)
#     std = [s if s > 1e-6 else 1.0 for s in std]
    
#     # 简单的 Min/Max 统计用于参考
#     min_val = np.min(all_data, axis=0).tolist()
#     max_val = np.max(all_data, axis=0).tolist()

#     stats = {
#         "action_mean": mean,
#         "action_std": std,
#         "action_min": min_val,
#         "action_max": max_val
#     }
    
#     print("=== Statistics Computed ===")
#     print(f"Mean: {mean}")
#     print(f"Std:  {std}")
    
#     with open(output_path, 'w') as f:
#         json.dump(stats, f, indent=4)
#     print(f"Saved stats to {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
#     parser.add_argument('--output_path', type=str, default='/yanghaochuan/projects/data/dataset_stats.json')
#     args = parser.parse_args()
#     compute_stats(args.data_root, args.output_path)

import h5py
with h5py.File("/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5", 'r') as f:
    print(f['data/demo_0/teacher_siglip'].shape) 
    # 必须输出 (T, 1152)。如果是 (T, 768)，你必须重新生成数据！