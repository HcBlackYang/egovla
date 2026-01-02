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

# def load_siglip_teacher(model_path, device):
#     print(f"正在加载 SigLIP 教师: {model_path}...")
#     try:
#         config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#         model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True).to(device).eval()
#         processor = AutoProcessor.from_pretrained(model_path)
#         print("SigLIP 加载成功！")
#         return model, processor
#     except Exception as e:
#         print(f"SigLIP 加载失败: {e}")
#         return None, None

# @torch.no_grad()
# def extract_teacher_features(model, processor, images, device, batch_size=32):
#     if model is None:
#         return np.zeros((len(images), 768), dtype=np.float32)
#     features_list = []
#     for i in range(0, len(images), batch_size):
#         batch_imgs = images[i : i + batch_size]
#         inputs = processor(images=list(batch_imgs), return_tensors="pt").to(device)
#         outputs = model.get_image_features(**inputs)
#         outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
#         features_list.append(outputs.cpu().numpy())
#     return np.concatenate(features_list, axis=0)

# def save_to_hdf5(raw_dir, output_path, siglip_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     siglip_model, siglip_processor = load_siglip_teacher(siglip_path, device)
    
#     episodes = sorted([os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
#     print(f"发现 {len(episodes)} 个演示数据，开始处理...")

#     with h5py.File(output_path, 'w') as f:
#         grp = f.create_group("data")
#         total_frames = 0
#         valid_episodes_count = 0

#         # 裁剪参数：头尾各切 90 帧 (3秒)
#         TRIM_HEAD = 90
#         TRIM_TAIL = 90
        
#         for i, ep_path in enumerate(tqdm(episodes)):
#             # 1. 读取视频
#             main_video = os.path.join(ep_path, "main_image.mp4")
#             if not os.path.exists(main_video): continue
#             main_imgs_raw = read_video_frames(main_video) 
            
#             wrist_video = os.path.join(ep_path, "wrist_image.mp4")
#             if os.path.exists(wrist_video):
#                 wrist_imgs_raw = read_video_frames(wrist_video)
#             else:
#                 wrist_imgs_raw = []

#             # 2. 读取状态
#             state_path = os.path.join(ep_path, "FrankaEmika_states.json")
#             if not os.path.exists(state_path): continue
#             qpos, gripper_width = read_json_state(state_path)
            
#             # === 3. 对齐 (Alignment) ===
#             target_len = len(main_imgs_raw)
#             current_len = len(qpos)
            
#             if current_len > target_len:
#                 indices = np.linspace(0, current_len - 1, target_len).astype(int)
#                 qpos = qpos[indices]
#                 if gripper_width is not None:
#                     gripper_width = gripper_width[indices]
#             else:
#                 target_len = min(current_len, target_len)
#                 main_imgs_raw = main_imgs_raw[:target_len]
#                 qpos = qpos[:target_len]
#                 if len(wrist_imgs_raw) > 0:
#                     wrist_imgs_raw = wrist_imgs_raw[:target_len]
#                 if gripper_width is not None:
#                     gripper_width = gripper_width[:target_len]

#             # === 4. 裁剪 (Trimming) ===
#             total_len = len(qpos)
#             start_idx = TRIM_HEAD
#             end_idx = total_len - TRIM_TAIL
            
#             if start_idx >= end_idx:
#                 continue
                
#             if (end_idx - start_idx) < 30: 
#                 continue

#             # === 5. 应用切片 ===
#             main_imgs_raw = main_imgs_raw[start_idx : end_idx]
#             qpos = qpos[start_idx : end_idx]
            
#             if len(wrist_imgs_raw) > 0:
#                 if len(wrist_imgs_raw) > end_idx:
#                     wrist_imgs_raw = wrist_imgs_raw[start_idx : end_idx]
#                 else:
#                     wrist_imgs_raw = wrist_imgs_raw[:total_len]
#                     if len(wrist_imgs_raw) > start_idx:
#                         wrist_imgs_raw = wrist_imgs_raw[start_idx : min(len(wrist_imgs_raw), end_idx)]
#                     else:
#                         wrist_imgs_raw = [] 

#             if gripper_width is not None:
#                 gripper_width = gripper_width[start_idx : end_idx]

#             # === 6. 拼接 Action (这是 8维向量) ===
#             if gripper_width is not None:
#                 if len(gripper_width.shape) == 1:
#                     gripper_width = gripper_width[:, np.newaxis]
#                 state_vec = np.hstack([qpos, gripper_width])
#             else:
#                 state_vec = qpos
            
#             current_len = len(state_vec)

#             # === Resize & Feature Extraction ===
#             main_imgs_resized = np.array([cv2.resize(img, (224, 224)) for img in main_imgs_raw], dtype=np.uint8)
            
#             if len(wrist_imgs_raw) > 0 and len(wrist_imgs_raw) == current_len:
#                 wrist_imgs_resized = np.array([cv2.resize(img, (224, 224)) for img in wrist_imgs_raw], dtype=np.uint8)
#             else:
#                 wrist_imgs_resized = np.zeros((current_len, 224, 224, 3), dtype=np.uint8)

#             teacher_siglip = extract_teacher_features(siglip_model, siglip_processor, main_imgs_raw, device)
            
#             if len(wrist_imgs_raw) > 0 and len(wrist_imgs_raw) == current_len:
#                 teacher_exo = extract_teacher_features(siglip_model, siglip_processor, wrist_imgs_raw, device)
#             else:
#                 teacher_exo = np.zeros_like(teacher_siglip)

#             # === Write HDF5 (关键修改点！) ===
#             ep_grp = grp.create_group(f"demo_{valid_episodes_count}") 
#             obs_grp = ep_grp.create_group("obs")
            
#             obs_grp.create_dataset("agentview_image", data=main_imgs_resized, compression="gzip")
#             obs_grp.create_dataset("robot0_eye_in_hand_image", data=wrist_imgs_resized, compression="gzip")
            
#             # --- 核心修改：这里必须存 state_vec (8维)，不能存 qpos (7维) ---
#             obs_grp.create_dataset("robot0_joint_pos", data=state_vec) 
#             # --------------------------------------------------------
            
#             ep_grp.create_dataset("teacher_siglip", data=teacher_siglip)
#             ep_grp.create_dataset("teacher_exo", data=teacher_exo)
#             ep_grp.create_dataset("actions", data=state_vec) 
            
#             task_path = os.path.join(ep_path, "task_info.json")
#             instruction = "pick up the paper cup"
#             if os.path.exists(task_path):
#                 try:
#                     with open(task_path, 'r') as tf:
#                         t_data = json.load(tf)
#                         instruction = t_data.get('instruction', t_data.get('task_description', instruction))
#                 except: pass
#             ep_grp.attrs["language_instruction"] = instruction
            
#             total_frames += current_len
#             valid_episodes_count += 1

#     print(f"处理完成！共保存 {valid_episodes_count} 条有效数据。")
#     print(f"HDF5已保存至: {output_path}")

# def read_video_frames(path):
#     frames = []
#     cap = cv2.VideoCapture(path)
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#         frames.append(frame)
#     cap.release()
#     return frames

# def read_json_state(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     states = []
#     grippers = []
    
#     if isinstance(data, list):
#         for s in data:
#             q = s.get('q', s.get('qpos', s.get('joint_positions', s.get('joint_angles', np.zeros(7)))))
#             if isinstance(q, list) or isinstance(q, np.ndarray):
#                 states.append(q[:7]) 
#             else:
#                 states.append(np.zeros(7))
            
#             g = s.get('gripper_width', s.get('width', s.get('gripper_qpos', 0.0)))
#             if isinstance(g, list) or isinstance(g, np.ndarray):
#                 grippers.append(g[0])
#             else:
#                 grippers.append(float(g))
                
#     return np.array(states, dtype=np.float32), np.array(grippers, dtype=np.float32)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--raw_dir', type=str, default='/yanghaochuan/data/pick_up_the_paper_cup', help='原始数据目录')
#     parser.add_argument('--out_path', type=str, default='/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5')
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
    
    # 确保按照时间/文件名排序，这样 B, A, A, A, A 的顺序才能对应上
    episodes = sorted([os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    print(f"发现 {len(episodes)} 个演示数据，开始处理...")

    with h5py.File(output_path, 'w') as f:
        grp = f.create_group("data")
        total_frames = 0
        valid_episodes_count = 0

        # 裁剪参数：头尾各切 90 帧 (3秒)
        TRIM_HEAD = 90
        TRIM_TAIL = 90
        
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

            # === 6. 拼接 Action (8维向量) ===
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

            # === Write HDF5 ===
            ep_grp = grp.create_group(f"demo_{valid_episodes_count}") 
            
            # [关键修改] 根据 5 的周期写入类型标记
            # 0, 5, 10... 是 Type B (Anchor)
            is_anchor = (valid_episodes_count % 5 == 0)
            ep_grp.attrs["data_type"] = "type_b" if is_anchor else "type_a"
            
            obs_grp = ep_grp.create_group("obs")
            
            obs_grp.create_dataset("agentview_image", data=main_imgs_resized, compression="gzip")
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=wrist_imgs_resized, compression="gzip")
            
            # 存 state_vec (8维)
            obs_grp.create_dataset("robot0_joint_pos", data=state_vec) 
            
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
    parser.add_argument('--raw_dir', type=str, default='/yanghaochuan/data/pick_up_the_orange_ball', help='原始数据目录')
    parser.add_argument('--out_path', type=str, default='/yanghaochuan/data/pick_up_the_orange_ball.hdf5')
    parser.add_argument('--siglip_path', type=str, default='/yanghaochuan/models/siglip-so400m-patch14-384')
    
    args = parser.parse_args()
    save_to_hdf5(args.raw_dir, args.out_path, args.siglip_path)