# # import torch
# # import numpy as np
# # from utils.dataset_loader import RobotDataset
# # import matplotlib.pyplot as plt

# # # 指向你的数据路径
# # DATA_PATH = '/yanghaochuan/data/12pick_up_the_orange_ball.hdf5'

# # def check_gripper():
# #     print("正在检查数据集夹爪数据...")
# #     ds = RobotDataset(hdf5_path=DATA_PATH, window_size=16, pred_horizon=64)
    
# #     gripper_variance_found = False
    
# #     # 随机抽查 10 个样本
# #     indices = np.random.choice(len(ds), 10)
    
# #     for i in indices:
# #         sample = ds[i]
# #         actions = sample['action_target'] # [64, 8]
        
# #         # 取第 8 维 (索引 7)
# #         gripper_vals = actions[:, 7].numpy()
        
# #         # 反归一化看看真实值 (假设 0是开, 1是关)
# #         mean_g = ds.action_mean[7].item()
# #         std_g = ds.action_std[7].item()
# #         raw_gripper = gripper_vals * std_g + mean_g
        
# #         print(f"Sample {i} Gripper Raw Range: [{raw_gripper.min():.4f}, {raw_gripper.max():.4f}]")
        
# #         if raw_gripper.max() - raw_gripper.min() > 0.1:
# #             gripper_variance_found = True
# #             print("✅ 发现夹爪动作变化！")
            
# #     if not gripper_variance_found:
# #         print("\n❌ 警告：抽样的样本中夹爪似乎没有动过！")
# #         print("这确认了 dataset_loader.py 没有正确读取夹爪数据。")
# #         print("请按上面的建议修改 dataset_loader.py 并重新训练。")
# #     else:
# #         print("\n✅ 数据集读取正常。问题可能出在推理代码。")

# # if __name__ == "__main__":
# #     check_gripper()

# import h5py
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import os
# import argparse

# def check_alignment(hdf5_path, demo_key='demo_0', output_dir='verify_output'):
#     """
#     全面检测 HDF5 中的一个 demo
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     print(f"🔍 正在检查文件: {hdf5_path} | 演示: {demo_key}")

#     try:
#         with h5py.File(hdf5_path, 'r') as f:
#             if 'data' not in f or demo_key not in f['data']:
#                 print(f"❌ 找不到 Key: {demo_key}。可用 Keys: {list(f['data'].keys())[:5]}...")
#                 return

#             demo_grp = f['data'][demo_key]
            
#             # === 1. 读取数据 ===
#             # 适配不同的 Key 命名习惯
#             main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
#             wrist_key = 'robot0_eye_in_hand_image'
            
#             # 读取视频 (假设是 N, H, W, C 或 N, C, H, W)
#             main_imgs = demo_grp['obs'][main_key][:]
#             wrist_imgs = demo_grp['obs'][wrist_key][:]
            
#             # 读取状态和动作
#             # 注意：Actions 通常比 Obs 少 1 帧或者一样，取决于你的生成逻辑
#             # 在你的项目中，Action 应该是预测下一步，长度通常和 Obs 一致
#             actions = demo_grp['actions'][:] 
#             robot_state = demo_grp['obs']['robot0_joint_pos'][:]
            
#             # 获取夹爪数据 (假设最后一维是夹爪)
#             # 如果是 Franka，Action 是 7+1=8 维，Obs 是 7+2=9 维 (包含 gripper width) 或 8 维
#             gripper_action = actions[:, -1]
#             gripper_state = robot_state[:, -1] # 或者是 joint_pos 的最后一位

#             # === 2. 基础长度检查 ===
#             len_img = len(main_imgs)
#             len_action = len(actions)
#             len_state = len(robot_state)
            
#             print(f"📊 数据长度检查:")
#             print(f"   - Video Frames: {len_img}")
#             print(f"   - Actions:      {len_action}")
#             print(f"   - States:       {len_state}")
            
#             if len_img != len_action or len_img != len_state:
#                 print(f"⚠️ 警告: 数据长度不一致！可能导致训练时索引越界。")
#             else:
#                 print(f"✅ 长度对齐通过。")

#             # # === 3. 裁剪逻辑可视化 (Gripper Curve) ===
#             plt.figure(figsize=(12, 4))
#             plt.plot(gripper_state, label='Gripper State (Width)', color='blue')
#             plt.plot(gripper_action, label='Gripper Action', color='orange', alpha=0.5, linestyle='--')
            
#             # 标记起点和终点
#             plt.axvline(x=0, color='green', linestyle=':', label='Start (Frame 0)')
#             plt.axvline(x=len_img-1, color='red', linestyle=':', label='End (Last Frame)')
            
#             plt.title(f'Gripper Width Analysis ({demo_key})')
#             plt.xlabel('Time Step')
#             plt.ylabel('Width / Signal')
#             plt.legend()
#             plt.grid(True)
            
#             curve_path = os.path.join(output_dir, f'{demo_key}_gripper_curve.png')
#             plt.savefig(curve_path)
#             print(f"📈 夹爪曲线已保存: {curve_path} (请检查曲线两端是否符合裁剪预期)")
#             plt.close()

#             # === 4. 视频对齐生成 (Visual Overlay) ===
#             print(f"🎥 正在生成验证视频 (Overlay)...")
#             video_save_path = os.path.join(output_dir, f'{demo_key}_verification.mp4')
            
#             # 假设图片是 (H, W, 3) 0-255，如果不是需要转换
#             # 你的 dataset_loader 里有 permute(2,0,1)，说明 HDF5 里存的可能是 HWC
#             # 我们先检查 shape
#             if main_imgs.shape[-1] != 3 and main_imgs.shape[1] == 3:
#                 # 是 N, C, H, W -> 转 N, H, W, C
#                 main_imgs = np.transpose(main_imgs, (0, 2, 3, 1))
#                 wrist_imgs = np.transpose(wrist_imgs, (0, 2, 3, 1))
            
#             H, W, _ = main_imgs[0].shape
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             # 拼接 Main 和 Wrist
#             out = cv2.VideoWriter(video_save_path, fourcc, 20.0, (W * 2, H))
            
#             for i in range(len_img):
#                 # 转换颜色 RGB -> BGR
#                 m_img = cv2.cvtColor(main_imgs[i], cv2.COLOR_RGB2BGR)
#                 w_img = cv2.cvtColor(wrist_imgs[i], cv2.COLOR_RGB2BGR)
                
#                 # 拼接
#                 combined = np.hstack([m_img, w_img])
                
#                 # 绘制数据条 (HUD)
#                 # 1. 夹爪数值
#                 g_val = gripper_state[i] if i < len(gripper_state) else 0
#                 g_act = gripper_action[i] if i < len(gripper_action) else 0
                
#                 # 2. 动作幅度 (主要关节速度的 L2 范数)
#                 # 假设前6维是手臂关节
#                 arm_action_norm = np.linalg.norm(actions[i, :6]) if i < len(actions) else 0
                
#                 # 在视频上写字
#                 text1 = f"Frame: {i}"
#                 text2 = f"Gripper State: {g_val:.4f}"
#                 text3 = f"Gripper Action: {g_act:.4f}"
#                 text4 = f"Arm Move: {arm_action_norm:.4f}"
                
#                 cv2.putText(combined, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                 cv2.putText(combined, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
#                 # 如果夹爪在动作，用红色高亮
#                 color_act = (0, 0, 255) if abs(g_act) > 0.5 else (200, 200, 200)
#                 cv2.putText(combined, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_act, 1)
                
#                 # 如果机械臂在动，用红色显示
#                 color_move = (0, 0, 255) if arm_action_norm > 0.1 else (200, 200, 200)
#                 cv2.putText(combined, text4, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_move, 1)

#                 out.write(combined)
            
#             out.release()
#             print(f"✅ 验证视频已生成: {video_save_path}")
#             print(f"👉 请下载视频查看：当画面中机械臂移动时，'Arm Move' 数值是否同步变大？")

#     except Exception as e:
#         print(f"❌ 检测出错: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', type=str, required=True, help='生成的 HDF5 文件路径')
#     parser.add_argument('--demo', type=str, default='demo_55', help='要检查的 Demo ID')
#     args = parser.parse_args()
    
#     check_alignment(args.path, args.demo)
import h5py
import numpy as np

# 将此处替换为您的 HDF5 文件路径
HDF5_PATH = "/yanghaochuan/data/hdf5/pick_up_the_orange_ball_and_put_it_on_the_plank.hdf5"

def check_frequency(path):
    try:
        with h5py.File(path, 'r') as f:
            print(f"📂 打开文件: {path}")
            
            # 1. 检查根目录属性 (Global Attributes)
            print("\n--- [1] 根目录属性 ---")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
                
            if 'data' not in f:
                print("❌ 错误: 文件中没有 'data' 组")
                return

            # 获取第一个演示 (Demo)
            demo_keys = list(f['data'].keys())
            if not demo_keys:
                print("❌ 错误: 'data' 组为空")
                return
                
            first_demo = f['data'][demo_keys[0]]
            print(f"\n--- [2] 演示 {demo_keys[0]} 属性 ---")
            for k, v in first_demo.attrs.items():
                print(f"  {k}: {v}")

            # 2. 尝试通过时间戳计算 (Time/Timestamp)
            # 常见的 key: 'time', 'timestamp', 'obs/timestamp'
            print("\n--- [3] 通过时间戳计算 ---")
            timestamps = None
            
            # 查找可能的时间戳位置
            if 'time' in first_demo:
                timestamps = first_demo['time'][:]
            elif 'timestamp' in first_demo:
                timestamps = first_demo['timestamp'][:]
            elif 'obs' in first_demo and 'timestamp' in first_demo['obs']:
                timestamps = first_demo['obs']['timestamp'][:]
            
            if timestamps is not None:
                # 计算相邻帧的时间差 (dt)
                dt = np.diff(timestamps)
                avg_dt = np.mean(dt)
                freq = 1.0 / avg_dt
                print(f"✅ 找到时间戳! 数据长度: {len(timestamps)}")
                print(f"  平均时间间隔 (dt): {avg_dt:.4f} 秒")
                print(f"  计算出的频率: {freq:.2f} Hz")
                print(f"  >> 建议填入配置的频率: {int(round(freq))}")
            else:
                print("⚠️ 未在常见位置找到时间戳 ('time', 'timestamp')。")
                print("   如果您的数据没有保存时间戳，请回想录制时的设置。")

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    check_frequency(HDF5_PATH)

# import h5py
# import glob
# import os

# # 1. 设定你的数据路径
# data_dir = "/yanghaochuan/data/hdf5" # 修改为你的文件夹路径
# file_pattern = "*.hdf5" # 或者 *.h5

# # 2. 设定正确的指令
# # 注意：你需要确保这个指令完全覆盖你之前的错误指令
# correct_instruction = "pick up the orange ball and put it on the plank"

# # 获取所有文件列表
# files = glob.glob(os.path.join(data_dir, file_pattern))
# print(f"Found {len(files)} files. Starting correction...")

# for file_path in files:
#     try:
#         # 使用 'r+' 模式打开，允许读写
#         with h5py.File(file_path, 'r+') as f:
            
#             # 情况 A: language_instruction 是一个 Dataset (数组)
#             # 很多数据加载器会将指令存为 bytes 格式的数组
#             if 'language_instruction' in f.keys():
#                 # 删除旧的 dataset
#                 del f['language_instruction']
                
#                 # 创建新的 dataset
#                 # 注意：HDF5 通常存储 numpy bytes 字符串
#                 dt = h5py.special_dtype(vlen=str) 
#                 f.create_dataset('language_instruction', data=correct_instruction, dtype=dt)
                
#             # 情况 B: language_instruction 是一个 Attribute (属性)
#             elif 'language_instruction' in f.attrs:
#                 f.attrs['language_instruction'] = correct_instruction
                
#             else:
#                 print(f"Warning: 'language_instruction' key not found in {os.path.basename(file_path)}")
                
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")

# print("Correction finished!")

# import h5py

# import h5py
# import glob
# import os

# # 1. 设置路径
# data_dir = "/yanghaochuan/data/hdf5"  # 请修改为你的实际路径
# file_pattern = "*.hdf5" 
# correct_instruction = "pick up the orange ball and put it on the plank"

# files = glob.glob(os.path.join(data_dir, file_pattern))
# print(f"找到 {len(files)} 个文件，准备开始修复...")

# for file_path in files:
#     try:
#         with h5py.File(file_path, 'r+') as f:
#             # 检查是否有 'data' 这个主组
#             if 'data' in f.keys():
#                 # 遍历 data 下面的所有 demo (例如 demo_0, demo_1, demo_43...)
#                 for demo_key in f['data'].keys():
#                     demo_group = f['data'][demo_key]
                    
#                     # 修改该 demo 组的属性
#                     if 'language_instruction' in demo_group.attrs:
#                         old_text = demo_group.attrs['language_instruction']
#                         demo_group.attrs['language_instruction'] = correct_instruction
#                         print(f"[{os.path.basename(file_path)}] {demo_key}: '{old_text}' -> '{correct_instruction}'")
#                     else:
#                         # 如果原本没有这个属性，也可以选择强制加上
#                         demo_group.attrs['language_instruction'] = correct_instruction
#                         print(f"[{os.path.basename(file_path)}] {demo_key}: 添加了新指令")
#             else:
#                 print(f"警告: {file_path} 中没有找到 'data' 组")

#     except Exception as e:
#         print(f"处理 {file_path} 时出错: {e}")

# print("所有文件修复完成！")