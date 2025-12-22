# # utils/preprocess_with_teachers.py
# import os
# import json
# import cv2
# import h5py
# import numpy as np
# import torch
# import argparse
# from tqdm import tqdm
# from transformers import AutoModel, AutoProcessor, AutoConfig


# # === 1. 新增：自动寻找动作起始和结束点的函数 ===
# def find_active_range(qpos, threshold=0.01):
#     """
#     根据关节角速度判断机器人的活跃范围。
#     qpos: [T, 7] or [T, 8] numpy array
#     threshold: 判定为运动的阈值 (弧度差)
#     """
#     # 计算相邻帧的关节变化量 (类似于速度)
#     delta = np.abs(qpos[1:] - qpos[:-1]).max(axis=1)
    
#     # 找到所有“动起来”的帧的索引
#     moving_indices = np.where(delta > threshold)[0]
    
#     if len(moving_indices) == 0:
#         print("[Warning] 该数据完全没有运动！建议丢弃。")
#         return 0, len(qpos)
    
#     # 起始点：第一次动起来的地方
#     # 建议多保留前 5-10 帧作为缓冲 (buffer)，让模型知道从静止到启动的过程
#     start_idx = max(0, moving_indices[0] - 5)
    
#     # 结束点：最后一次动的地方
#     end_idx = min(len(qpos), moving_indices[-1] + 10)
    
#     return start_idx, end_idx

# def load_siglip_teacher(model_path, device):
#     """加载 SigLIP 教师模型"""
#     print(f"正在加载 SigLIP 教师: {model_path}...")
#     try:
#         # 尝试加载本地模型
#         config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#         model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).to(device).eval()
#         processor = AutoProcessor.from_pretrained(model_path)
#         print("SigLIP 加载成功！")
#         return model, processor
#     except Exception as e:
#         print(f"SigLIP 加载失败: {e}")
#         print("请确保你已下载 'google/siglip-so400m-patch14-384' 并指定了正确路径。")
#         return None, None

# @torch.no_grad()
# def extract_teacher_features(model, processor, images, device, batch_size=32):
#     """批量提取教师特征"""
#     if model is None:
#         # 如果没有教师，返回全0占位符 (仅用于调试流程，实际训练还是无效)
#         return np.zeros((len(images), 768), dtype=np.float32) # 假设维度768，视具体模型而定

#     features_list = []
    
#     # 简单的批量处理
#     for i in range(0, len(images), batch_size):
#         batch_imgs = images[i : i + batch_size]
        
#         # SigLIP 预处理
#         # images 是 [H, W, C] list, processor 期望 PIL 或 numpy array
#         inputs = processor(images=list(batch_imgs), return_tensors="pt").to(device)
        
#         # 推理 (获取 image_embeds)
#         outputs = model.get_image_features(**inputs)
        
#         # 归一化 (SigLIP/CLIP 特征通常需要归一化)
#         outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
#         features_list.append(outputs.cpu().numpy())

#     return np.concatenate(features_list, axis=0)

# # def save_to_hdf5(raw_dir, output_path, siglip_path):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# #     # 1. 加载教师
# #     siglip_model, siglip_processor = load_siglip_teacher(siglip_path, device)
    
# #     # 2. 扫描数据
# #     # 根据你的截图，数据在 raw_dir/2025...
# #     episodes = sorted([os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
# #     print(f"发现 {len(episodes)} 个演示数据，开始处理...")

# #     with h5py.File(output_path, 'w') as f:
# #         grp = f.create_group("data")
# #         total_frames = 0
        
# #         for i, ep_path in enumerate(tqdm(episodes)):
# #             ep_grp = grp.create_group(f"demo_{i}")
# #             obs_grp = ep_grp.create_group("obs")
            
# #             # --- 读取基础数据 ---
# #             # Main Video
# #             main_video = os.path.join(ep_path, "main_image.mp4")
# #             main_imgs_raw = read_video_frames(main_video) # 保持原始分辨率用于教师提取
# #             main_imgs_resized = [cv2.resize(img, (224, 224)) for img in main_imgs_raw] # 用于存入HDF5
            
# #             # Wrist Video
# #             wrist_video = os.path.join(ep_path, "wrist_image.mp4")
# #             if os.path.exists(wrist_video):
# #                 wrist_imgs_raw = read_video_frames(wrist_video)
# #                 wrist_imgs_resized = [cv2.resize(img, (224, 224)) for img in wrist_imgs_raw]
# #             else:
# #                 wrist_imgs_resized = np.zeros_like(main_imgs_resized)
# #                 wrist_imgs_raw = []

# #             # State
# #             state_path = os.path.join(ep_path, "FrankaEmika_states.json")
# #             qpos = read_json_state(state_path)
            
# #             # 对齐长度
# #             min_len = min(len(main_imgs_resized), len(qpos))
# #             if len(wrist_imgs_resized) > 0 and len(wrist_imgs_raw) > 0:
# #                 min_len = min(min_len, len(wrist_imgs_resized))
            
# #             # 截断
# #             main_imgs_raw = main_imgs_raw[:min_len]
# #             main_imgs_resized = np.array(main_imgs_resized[:min_len], dtype=np.uint8)
# #             wrist_imgs_resized = np.array(wrist_imgs_resized[:min_len], dtype=np.uint8)
# #             qpos = qpos[:min_len]

# #             # --- 核心：提取教师特征 ---
# #             # SigLIP 看着 Main Image
# #             teacher_siglip = extract_teacher_features(siglip_model, siglip_processor, main_imgs_raw, device)
            
# #             # Exo Teacher (暂时用全0或Wrist的SigLIP特征代替，如果你没有专门的手部模型)
# #             # 这里我们为了跑通代码，暂时先用 SigLIP 提取 wrist 图像作为 "exo_features" 的替代
# #             # 这比全0要好，至少代表了手眼视角的语义
# #             if len(wrist_imgs_raw) > 0:
# #                 teacher_exo = extract_teacher_features(siglip_model, siglip_processor, wrist_imgs_raw[:min_len], device)
# #             else:
# #                 teacher_exo = np.zeros_like(teacher_siglip)

# #             # --- 写入 HDF5 ---
# #             obs_grp.create_dataset("agentview_image", data=main_imgs_resized, compression="gzip")
# #             obs_grp.create_dataset("robot0_eye_in_hand_image", data=wrist_imgs_resized, compression="gzip")
# #             obs_grp.create_dataset("robot0_joint_pos", data=qpos)
            
# #             # 写入教师特征 (不需要压缩，这是浮点数)
# #             # 注意：Key名要和 dataset_loader 里的读取逻辑对应
# #             ep_grp.create_dataset("teacher_siglip", data=teacher_siglip)
# #             ep_grp.create_dataset("teacher_exo", data=teacher_exo)
            
# #             ep_grp.create_dataset("actions", data=qpos)
            
# #             # 读取文本
# #             task_path = os.path.join(ep_path, "task_info.json")
# #             instruction = "do something"
# #             if os.path.exists(task_path):
# #                 try:
# #                     with open(task_path, 'r') as tf:
# #                         t_data = json.load(tf)
# #                         instruction = t_data.get('instruction', t_data.get('task_description', instruction))
# #                 except: pass
# #             ep_grp.attrs["language_instruction"] = instruction
            
# #             total_frames += min_len

# #     print(f"处理完成！HDF5已保存至: {output_path}")

# def save_to_hdf5(raw_dir, output_path, siglip_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 1. 加载教师模型
#     siglip_model, siglip_processor = load_siglip_teacher(siglip_path, device)
    
#     # 2. 扫描数据
#     episodes = sorted([os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
#     print(f"发现 {len(episodes)} 个演示数据，开始处理...")

#     with h5py.File(output_path, 'w') as f:
#         grp = f.create_group("data")
#         total_frames = 0
#         valid_episodes_count = 0
        
#         for i, ep_path in enumerate(tqdm(episodes)):
#             # --- 读取基础数据 ---
#             # Main Video
#             main_video = os.path.join(ep_path, "main_image.mp4")
#             if not os.path.exists(main_video):
#                 continue
#             main_imgs_raw = read_video_frames(main_video) 
            
#             # Wrist Video
#             wrist_video = os.path.join(ep_path, "wrist_image.mp4")
#             if os.path.exists(wrist_video):
#                 wrist_imgs_raw = read_video_frames(wrist_video)
#             else:
#                 wrist_imgs_raw = []

#             # State
#             state_path = os.path.join(ep_path, "FrankaEmika_states.json")
#             if not os.path.exists(state_path):
#                 continue
#             qpos = read_json_state(state_path)
            
#             # 基础对齐长度 (防止视频和json长度不一致)
#             min_len = min(len(main_imgs_raw), len(qpos))
#             if len(wrist_imgs_raw) > 0:
#                 min_len = min(min_len, len(wrist_imgs_raw))
            
#             # 第一次截断
#             main_imgs_raw = main_imgs_raw[:min_len]
#             if len(wrist_imgs_raw) > 0:
#                 wrist_imgs_raw = wrist_imgs_raw[:min_len]
#             qpos = qpos[:min_len]

#             # === 核心修改：自动切除静止帧 ===
#             start_idx, end_idx = find_active_range(qpos)
            
#             # 如果数据无效 (完全不动) 或 切割后太短 (小于窗口大小16)，则跳过
#             if start_idx is None or (end_idx - start_idx) < 16:
#                 print(f"Skipping episode {i}: Too short or stationary.")
#                 continue

#             # 应用切片 (只保留动起来的部分)
#             main_imgs_raw = main_imgs_raw[start_idx : end_idx]
#             qpos = qpos[start_idx : end_idx]
#             if len(wrist_imgs_raw) > 0:
#                 wrist_imgs_raw = wrist_imgs_raw[start_idx : end_idx]
            
#             # 更新长度
#             current_len = len(qpos)
#             # ==============================

#             # Resize (现在只处理有效帧，速度更快)
#             main_imgs_resized = [cv2.resize(img, (224, 224)) for img in main_imgs_raw]
            
#             if len(wrist_imgs_raw) > 0:
#                 wrist_imgs_resized = [cv2.resize(img, (224, 224)) for img in wrist_imgs_raw]
#             else:
#                 # 如果没有手腕视频，创建全黑占位符
#                 wrist_imgs_resized = np.zeros((current_len, 224, 224, 3), dtype=np.uint8)

#             # 转为 Numpy
#             main_imgs_resized = np.array(main_imgs_resized, dtype=np.uint8)
#             wrist_imgs_resized = np.array(wrist_imgs_resized, dtype=np.uint8)

#             # --- 提取教师特征 (使用切片后的原始分辨率图像) ---
#             # SigLIP 看着 Main Image
#             teacher_siglip = extract_teacher_features(siglip_model, siglip_processor, main_imgs_raw, device)
            
#             # Exo Teacher (暂时用 SigLIP 提取手腕图像代替)
#             if len(wrist_imgs_raw) > 0:
#                 teacher_exo = extract_teacher_features(siglip_model, siglip_processor, wrist_imgs_raw, device)
#             else:
#                 teacher_exo = np.zeros_like(teacher_siglip)

#             # --- 写入 HDF5 ---
#             ep_grp = grp.create_group(f"demo_{valid_episodes_count}") # 使用新的计数索引
#             obs_grp = ep_grp.create_group("obs")

#             obs_grp.create_dataset("agentview_image", data=main_imgs_resized, compression="gzip")
#             obs_grp.create_dataset("robot0_eye_in_hand_image", data=wrist_imgs_resized, compression="gzip")
#             obs_grp.create_dataset("robot0_joint_pos", data=qpos)
            
#             ep_grp.create_dataset("teacher_siglip", data=teacher_siglip)
#             ep_grp.create_dataset("teacher_exo", data=teacher_exo)
#             ep_grp.create_dataset("actions", data=qpos)
            
#             # 读取文本
#             task_path = os.path.join(ep_path, "task_info.json")
#             instruction = "pick up the paper cup" # 默认指令
#             if os.path.exists(task_path):
#                 try:
#                     with open(task_path, 'r') as tf:
#                         t_data = json.load(tf)
#                         instruction = t_data.get('instruction', t_data.get('task_description', instruction))
#                 except: pass
#             ep_grp.attrs["language_instruction"] = instruction
            
#             total_frames += current_len
#             valid_episodes_count += 1

#     print(f"处理完成！共保存 {valid_episodes_count} 条有效数据 (原始 {len(episodes)} 条)。")
#     print(f"HDF5已保存至: {output_path}")

# def read_video_frames(path):
#     frames = []
#     cap = cv2.VideoCapture(path)
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Keep original resolution for Teacher
#         frames.append(frame)
#     cap.release()
#     return frames

# def read_json_state(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     states = []
#     if isinstance(data, list):
#         for s in data:
#             q = s.get('q', s.get('qpos', s.get('joint_positions', np.zeros(8))))
#             states.append(q[:8])
#     return np.array(states, dtype=np.float32)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # 你的原始数据路径
#     parser.add_argument('--raw_dir', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup', help='原始数据目录')
#     # 输出文件
#     parser.add_argument('--out_path', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
#     # SigLIP 模型路径
#     parser.add_argument('--siglip_path', type=str, default='/yanghaochuan/models/siglip-so400m-patch14-384')
    
#     args = parser.parse_args()
#     save_to_hdf5(args.raw_dir, args.out_path, args.siglip_path)

# utils/preprocess_with_teachers.py
import os
import json
import cv2
import h5py
import numpy as np
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoConfig

def load_siglip_teacher(model_path, device):
    print(f"正在加载 SigLIP 教师: {model_path}...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        print("SigLIP 加载成功！")
        return model, processor
    except Exception as e:
        print(f"SigLIP 加载失败: {e}")
        return None, None

@torch.no_grad()
def extract_teacher_features(model, processor, images, device, batch_size=32):
    if model is None:
        return np.zeros((len(images), 768), dtype=np.float32)
    features_list = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        inputs = processor(images=list(batch_imgs), return_tensors="pt").to(device)
        outputs = model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        features_list.append(outputs.cpu().numpy())
    return np.concatenate(features_list, axis=0)

def save_to_hdf5(raw_dir, output_path, siglip_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siglip_model, siglip_processor = load_siglip_teacher(siglip_path, device)
    
    episodes = sorted([os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    print(f"发现 {len(episodes)} 个演示数据，开始处理...")

    with h5py.File(output_path, 'w') as f:
        grp = f.create_group("data")
        total_frames = 0
        valid_episodes_count = 0
        
        # 裁剪参数：头尾各切 60 帧 (2秒)
        TRIM_HEAD = 60
        TRIM_TAIL = 60
        
        for i, ep_path in enumerate(tqdm(episodes)):
            # 1. 读取视频
            main_video = os.path.join(ep_path, "main_image.mp4")
            if not os.path.exists(main_video): continue
            main_imgs_raw = read_video_frames(main_video) 
            
            wrist_video = os.path.join(ep_path, "wrist_image.mp4")
            if os.path.exists(wrist_video):
                wrist_imgs_raw = read_video_frames(wrist_video)
            else:
                wrist_imgs_raw = []

            # 2. 读取状态
            state_path = os.path.join(ep_path, "FrankaEmika_states.json")
            if not os.path.exists(state_path): continue
            qpos, gripper_width = read_json_state(state_path)
            
            # === 3. 对齐 (Alignment) ===
            target_len = len(main_imgs_raw)
            current_len = len(qpos)
            
            if current_len > target_len:
                indices = np.linspace(0, current_len - 1, target_len).astype(int)
                qpos = qpos[indices]
                if gripper_width is not None:
                    gripper_width = gripper_width[indices]
            else:
                target_len = min(current_len, target_len)
                main_imgs_raw = main_imgs_raw[:target_len]
                qpos = qpos[:target_len]
                if len(wrist_imgs_raw) > 0:
                    wrist_imgs_raw = wrist_imgs_raw[:target_len]
                if gripper_width is not None:
                    gripper_width = gripper_width[:target_len]

            # === 4. 裁剪 (Trimming) ===
            total_len = len(qpos)
            start_idx = TRIM_HEAD
            end_idx = total_len - TRIM_TAIL
            
            if start_idx >= end_idx:
                continue
                
            if (end_idx - start_idx) < 30: 
                continue

            # === 5. 应用切片 ===
            main_imgs_raw = main_imgs_raw[start_idx : end_idx]
            qpos = qpos[start_idx : end_idx]
            
            if len(wrist_imgs_raw) > 0:
                if len(wrist_imgs_raw) > end_idx:
                    wrist_imgs_raw = wrist_imgs_raw[start_idx : end_idx]
                else:
                    wrist_imgs_raw = wrist_imgs_raw[:total_len]
                    if len(wrist_imgs_raw) > start_idx:
                        wrist_imgs_raw = wrist_imgs_raw[start_idx : min(len(wrist_imgs_raw), end_idx)]
                    else:
                        wrist_imgs_raw = [] 

            if gripper_width is not None:
                gripper_width = gripper_width[start_idx : end_idx]

            # === 6. 拼接 Action (这是 8维向量) ===
            if gripper_width is not None:
                if len(gripper_width.shape) == 1:
                    gripper_width = gripper_width[:, np.newaxis]
                state_vec = np.hstack([qpos, gripper_width])
            else:
                state_vec = qpos
            
            current_len = len(state_vec)

            # === Resize & Feature Extraction ===
            main_imgs_resized = np.array([cv2.resize(img, (224, 224)) for img in main_imgs_raw], dtype=np.uint8)
            
            if len(wrist_imgs_raw) > 0 and len(wrist_imgs_raw) == current_len:
                wrist_imgs_resized = np.array([cv2.resize(img, (224, 224)) for img in wrist_imgs_raw], dtype=np.uint8)
            else:
                wrist_imgs_resized = np.zeros((current_len, 224, 224, 3), dtype=np.uint8)

            teacher_siglip = extract_teacher_features(siglip_model, siglip_processor, main_imgs_raw, device)
            
            if len(wrist_imgs_raw) > 0 and len(wrist_imgs_raw) == current_len:
                teacher_exo = extract_teacher_features(siglip_model, siglip_processor, wrist_imgs_raw, device)
            else:
                teacher_exo = np.zeros_like(teacher_siglip)

            # === Write HDF5 (关键修改点！) ===
            ep_grp = grp.create_group(f"demo_{valid_episodes_count}") 
            obs_grp = ep_grp.create_group("obs")
            
            obs_grp.create_dataset("agentview_image", data=main_imgs_resized, compression="gzip")
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=wrist_imgs_resized, compression="gzip")
            
            # --- 核心修改：这里必须存 state_vec (8维)，不能存 qpos (7维) ---
            obs_grp.create_dataset("robot0_joint_pos", data=state_vec) 
            # --------------------------------------------------------
            
            ep_grp.create_dataset("teacher_siglip", data=teacher_siglip)
            ep_grp.create_dataset("teacher_exo", data=teacher_exo)
            ep_grp.create_dataset("actions", data=state_vec) 
            
            task_path = os.path.join(ep_path, "task_info.json")
            instruction = "pick up the paper cup"
            if os.path.exists(task_path):
                try:
                    with open(task_path, 'r') as tf:
                        t_data = json.load(tf)
                        instruction = t_data.get('instruction', t_data.get('task_description', instruction))
                except: pass
            ep_grp.attrs["language_instruction"] = instruction
            
            total_frames += current_len
            valid_episodes_count += 1

    print(f"处理完成！共保存 {valid_episodes_count} 条有效数据。")
    print(f"HDF5已保存至: {output_path}")

def read_video_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frames.append(frame)
    cap.release()
    return frames

def read_json_state(path):
    with open(path, 'r') as f:
        data = json.load(f)
    states = []
    grippers = []
    
    if isinstance(data, list):
        for s in data:
            q = s.get('q', s.get('qpos', s.get('joint_positions', s.get('joint_angles', np.zeros(7)))))
            if isinstance(q, list) or isinstance(q, np.ndarray):
                states.append(q[:7]) 
            else:
                states.append(np.zeros(7))
            
            g = s.get('gripper_width', s.get('width', s.get('gripper_qpos', 0.0)))
            if isinstance(g, list) or isinstance(g, np.ndarray):
                grippers.append(g[0])
            else:
                grippers.append(float(g))
                
    return np.array(states, dtype=np.float32), np.array(grippers, dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup', help='原始数据目录')
    parser.add_argument('--out_path', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
    parser.add_argument('--siglip_path', type=str, default='/yanghaochuan/models/siglip-so400m-patch14-384')
    
    args = parser.parse_args()
    save_to_hdf5(args.raw_dir, args.out_path, args.siglip_path)