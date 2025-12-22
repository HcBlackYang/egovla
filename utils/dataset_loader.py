# # utils/dataset_loader.py
# import torch
# from torch.utils.data import Dataset
# import h5py
# import numpy as np
# import os
# from transformers import T5Tokenizer

# # Franka 机械臂关节大致在 -2.9 到 2.9 之间，我们用 3.0 做归一化
# MAX_JOINT_RAD = 3.0 

# class RobotDataset(Dataset):
#     def __init__(self, hdf5_path, window_size=16, 
#                  tokenizer_path="/yanghaochuan/models/flan-t5-large"):
        
#         self.hdf5_path = hdf5_path
#         self.window_size = window_size
        
#         print(f"[Dataset] Loading Tokenizer from {tokenizer_path}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
#         except:
#             print("[Dataset] Local tokenizer failed, trying default...")
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
#         self.indices = []
#         with h5py.File(hdf5_path, 'r') as f:
#             if 'data' not in f:
#                  raise ValueError(f"HDF5结构错误: {hdf5_path} 中没有 'data' 组")

#             self.demos = list(f['data'].keys())
#             for demo_key in self.demos:
#                 try:
#                     demo_grp = f['data'][demo_key]
#                     if 'actions' not in demo_grp: continue
                    
#                     demo_len = demo_grp['actions'].shape[0]
#                     has_teacher = 'teacher_siglip' in demo_grp
                    
#                     # 我们需要读取 window_size + 1 (用于预测下一步)，所以总长度要够
#                     if demo_len > window_size:
#                         instruction = demo_grp.attrs.get('language_instruction', 'do nothing')
#                         # 这里的循环长度要减去 1，保证能取到下一步
#                         for i in range(demo_len - window_size): 
#                             self.indices.append({
#                                 'demo_key': demo_key,
#                                 'start_idx': i,
#                                 'instruction': instruction,
#                                 'has_teacher': has_teacher
#                             })
#                 except Exception as e:
#                     print(f"Skipping {demo_key}: {e}")
        
#         print(f"[Dataset] Loaded {len(self.indices)} samples from {os.path.basename(hdf5_path)}")

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         meta = self.indices[idx]
#         demo_key = meta['demo_key']
#         start = meta['start_idx']
        
#         with h5py.File(self.hdf5_path, 'r') as f:
#             demo_grp = f['data'][demo_key]
            
#             # --- 1. Video (保持 window_size 长度) ---
#             # [T, H, W, C] -> [T, C, H, W]
#             img_seq = demo_grp['obs']['robot0_eye_in_hand_image'][start : start + self.window_size]
#             video = torch.tensor(img_seq).float().permute(0, 3, 1, 2) / 255.0
            
#             # --- 2. State & Action Target (读取 window_size + 1) ---
#             # 我们多读一帧，前16帧是输入，第17帧是预测目标
#             state_seq_raw = demo_grp['obs']['robot0_joint_pos'][start : start + self.window_size + 1]
            
#             # 如果读出来的长度不够 (比如到了数据末尾)，用最后一帧补齐
#             if state_seq_raw.shape[0] < self.window_size + 1:
#                 pad_len = (self.window_size + 1) - state_seq_raw.shape[0]
#                 last_frame = state_seq_raw[-1:]
#                 state_seq_raw = np.concatenate([state_seq_raw, np.tile(last_frame, (pad_len, 1))], axis=0)

#             # === 归一化 ===
#             state_seq_norm = torch.tensor(state_seq_raw).float() / MAX_JOINT_RAD
#             # 前 7 维 (关节): 除以 3.0
#             state_seq_norm[:, :7] = state_seq_norm[:, :7] / 3.0
            
#             # 第 8 维 (夹爪): 如果数值很小(米单位)，稍微放大一点以便模型学习
#             # 假设原始是 0~0.08，我们乘以 10 变成 0~0.8，这样跟关节的幅度(0~1)就匹配了
#             if state_seq_norm.shape[-1] == 8:
#                 state_seq_norm[:, 7] = state_seq_norm[:, 7] * 10.0
                
#             # 切分输入和目标
#             state_input = state_seq_norm[:self.window_size]   # T=0~15
#             action_target = state_seq_norm[self.window_size]  # T=16 (Next Step)

#             # --- 3. First Frame ---
#             first_img = demo_grp['obs']['robot0_eye_in_hand_image'][0]
            
#             # --- 4. Teacher Features (保持 window_size 长度) ---
#             if meta['has_teacher']:
#                 teacher_siglip = torch.tensor(demo_grp['teacher_siglip'][start : start + self.window_size]).float()
#                 teacher_exo = torch.tensor(demo_grp['teacher_exo'][start : start + self.window_size]).float()
#             else:
#                 teacher_siglip = torch.zeros(self.window_size, 1152)
#                 teacher_exo = torch.zeros(self.window_size, 1152)

#         # Tokenize
#         cmd = meta['instruction']
#         if isinstance(cmd, bytes): cmd = cmd.decode('utf-8')
#         text_tokens = self.tokenizer(
#             cmd, return_tensors="pt", padding="max_length", max_length=16, truncation=True
#         ).input_ids.squeeze(0)
        
#         first_frame = torch.tensor(first_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

#         return {
#             "video": video,
#             "state": state_input,       # 输入给 Encoder
#             "action_target": action_target, # 预测目标 (Next Step)
#             "text_tokens": text_tokens,
#             "first_frame": first_frame,
#             "teacher_siglip": teacher_siglip,
#             "teacher_exo": teacher_exo
#         }

# utils/dataset_loader.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import json
from transformers import T5Tokenizer

class RobotDataset(Dataset):
    def __init__(self, hdf5_path, window_size=16, 
                 tokenizer_path="/yanghaochuan/models/flan-t5-large",
                 stats_path="/yanghaochuan/projects/data/dataset_stats.json"): # 默认指向你的统计文件
        
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        
        # === 1. 加载 Tokenizer ===
        print(f"[Dataset] Loading Tokenizer from {tokenizer_path}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except:
            print("[Dataset] Local tokenizer failed, trying default...")
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
        # === 2. 加载统计量 (核心修改) ===
        if not os.path.exists(stats_path):
             raise FileNotFoundError(f"❌ 找不到统计文件: {stats_path}。请先运行 utils/compute_stats.py！")
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # 转为 Tensor
        self.action_mean = torch.tensor(stats['action_mean']).float()
        self.action_std = torch.tensor(stats['action_std']).float()
        
        # [安全措施] 防止 std 太小导致除法爆炸
        # 如果某个关节几乎不动(std < 0.01)，我们就把它的 std 设为 1.0 (不缩放)，或者设为一个最小值 0.01
        # 这里为了保留物理意义，如果 std 极小，说明它是静止的，我们给一个阈值防止除以0
        self.action_std = torch.maximum(self.action_std, torch.tensor(1e-2))
        
        print(f"[Dataset] Loaded normalization stats.")
        print(f"   - Mean: {self.action_mean}")
        print(f"   - Std (Clamped): {self.action_std}")

        # === 3. 扫描数据 ===
        self.indices = []
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                 raise ValueError(f"HDF5结构错误: {hdf5_path} 中没有 'data' 组")

            self.demos = list(f['data'].keys())
            for demo_key in self.demos:
                try:
                    demo_grp = f['data'][demo_key]
                    if 'actions' not in demo_grp: continue
                    
                    demo_len = demo_grp['actions'].shape[0]
                    has_teacher = 'teacher_siglip' in demo_grp
                    
                    if demo_len > window_size:
                        instruction = demo_grp.attrs.get('language_instruction', 'do nothing')
                        # 保证有 window_size + 1 (用于预测下一步)
                        for i in range(demo_len - window_size): 
                            self.indices.append({
                                'demo_key': demo_key,
                                'start_idx': i,
                                'instruction': instruction,
                                'has_teacher': has_teacher
                            })
                except Exception as e:
                    print(f"Skipping {demo_key}: {e}")
        
        print(f"[Dataset] Loaded {len(self.indices)} samples from {os.path.basename(hdf5_path)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        meta = self.indices[idx]
        demo_key = meta['demo_key']
        start = meta['start_idx']
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_grp = f['data'][demo_key]
            
            # --- 1. Video ---
            img_seq = demo_grp['obs']['robot0_eye_in_hand_image'][start : start + self.window_size]
            video = torch.tensor(img_seq).float().permute(0, 3, 1, 2) / 255.0
            
            # --- 2. State & Action (读取 Window + 1) ---
            state_seq_raw = demo_grp['obs']['robot0_joint_pos'][start : start + self.window_size + 1]
            
            # 补齐逻辑
            if state_seq_raw.shape[0] < self.window_size + 1:
                pad_len = (self.window_size + 1) - state_seq_raw.shape[0]
                last_frame = state_seq_raw[-1:]
                state_seq_raw = np.concatenate([state_seq_raw, np.tile(last_frame, (pad_len, 1))], axis=0)

            # === 核心修改：Z-Score 归一化 ===
            state_seq_tensor = torch.tensor(state_seq_raw).float()
            
            # Formula: (X - Mean) / Std
            state_seq_norm = (state_seq_tensor - self.action_mean) / self.action_std
            
            # 切分
            state_input = state_seq_norm[:self.window_size]   # T=0~15
            action_target = state_seq_norm[self.window_size]  # T=16 (Next Step)

            # --- 3. First Frame ---
            first_img = demo_grp['obs']['robot0_eye_in_hand_image'][0]
            
            # --- 4. Teachers ---
            if meta['has_teacher']:
                teacher_siglip = torch.tensor(demo_grp['teacher_siglip'][start : start + self.window_size]).float()
                teacher_exo = torch.tensor(demo_grp['teacher_exo'][start : start + self.window_size]).float()
            else:
                teacher_siglip = torch.zeros(self.window_size, 1152)
                teacher_exo = torch.zeros(self.window_size, 1152)

        # Tokenize
        cmd = meta['instruction']
        if isinstance(cmd, bytes): cmd = cmd.decode('utf-8')
        text_tokens = self.tokenizer(
            cmd, return_tensors="pt", padding="max_length", max_length=16, truncation=True
        ).input_ids.squeeze(0)
        
        first_frame = torch.tensor(first_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        return {
            "video": video,
            "state": state_input,       
            "action_target": action_target, 
            "text_tokens": text_tokens,
            "first_frame": first_frame,
            "teacher_siglip": teacher_siglip,
            "teacher_exo": teacher_exo
        }