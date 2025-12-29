# utils/dataset_loader.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import json
from transformers import T5Tokenizer

class RobotDataset(Dataset):
    def __init__(self, hdf5_path, 
                 window_size=16, 
                 pred_horizon=16, # <--- 新增: 预测未来步数 (Chunk Size)
                 tokenizer_path="/yanghaochuan/models/flan-t5-large",
                 stats_path="/yanghaochuan/data/1223dataset_stats.json"): 
        
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        
        # === 1. 加载 Tokenizer ===
        print(f"[Dataset] Loading Tokenizer from {tokenizer_path}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except:
            print("[Dataset] Local tokenizer failed, trying default...")
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
        # === 2. 加载统计量 (用于 Z-Score 归一化) ===
        if not os.path.exists(stats_path):
             raise FileNotFoundError(f"❌ 找不到统计文件: {stats_path}。请先运行 utils/compute_stats.py！")
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.action_mean = torch.tensor(stats['action_mean']).float()
        self.action_std = torch.tensor(stats['action_std']).float()
        
        # [安全措施] 防止 std 太小导致除法爆炸
        self.action_std = torch.maximum(self.action_std, torch.tensor(1e-2))
        
        print(f"[Dataset] Loaded normalization stats.")

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
                    
                    total_len = demo_grp['actions'].shape[0]
                    has_teacher = 'teacher_siglip' in demo_grp
                    
                    # 确保长度足够: 历史窗口(16) + 预测窗口(16)
                    min_len = window_size + pred_horizon
                    
                    if total_len > min_len:
                        instruction = demo_grp.attrs.get('language_instruction', 'do nothing')
                        # 遍历每一个可能的起始点
                        for i in range(total_len - min_len): 
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
        
        # 计算读取长度: 历史16帧 + 未来16帧
        read_len = self.window_size + self.pred_horizon
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_grp = f['data'][demo_key]
            
            # --- 1. Video (读取双摄) ---
            # 假设主摄 key 为 'agentview_image'，手腕 key 为 'robot0_eye_in_hand_image'
            # 如果您的 key 不一样，请在这里修改
            main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
            wrist_key = 'robot0_eye_in_hand_image'
            
            # 读取主摄
            main_seq = demo_grp['obs'][main_key][start : start + self.window_size]
            main_tensor = torch.tensor(main_seq).float().permute(0, 3, 1, 2) / 255.0 # [T, 3, H, W]
            
            # 读取手腕
            wrist_seq = demo_grp['obs'][wrist_key][start : start + self.window_size]
            wrist_tensor = torch.tensor(wrist_seq).float().permute(0, 3, 1, 2) / 255.0 # [T, 3, H, W]
            
            # 堆叠双摄: [2, T, 3, H, W] -> permute -> [2, 3, T, H, W]
            # 这里的顺序 View 0 是 Main, View 1 是 Wrist
            video = torch.stack([main_tensor, wrist_tensor], dim=0).permute(0, 2, 1, 3, 4)
            
            # --- 2. State & Action (读取 Chunk) ---
            state_seq_raw = demo_grp['obs']['robot0_joint_pos'][start : start + read_len]
            
            # 补齐逻辑
            if state_seq_raw.shape[0] < read_len:
                pad_len = read_len - state_seq_raw.shape[0]
                last_frame = state_seq_raw[-1:]
                state_seq_raw = np.concatenate([state_seq_raw, np.tile(last_frame, (pad_len, 1))], axis=0)

            # Z-Score 归一化
            state_seq_tensor = torch.tensor(state_seq_raw).float()
            state_seq_norm = (state_seq_tensor - self.action_mean) / self.action_std
            
            # 切分: Input (历史) vs Target (未来 Chunk)
            state_input = state_seq_norm[:self.window_size]  # [16, 8]
            action_target = state_seq_norm[self.window_size : self.window_size + self.pred_horizon] # [16, 8]

            # --- 3. First Frame (双摄) ---
            main_first = demo_grp['obs'][main_key][0]
            wrist_first = demo_grp['obs'][wrist_key][0]
            
            main_first_t = torch.tensor(main_first).float().permute(2, 0, 1) / 255.0
            wrist_first_t = torch.tensor(wrist_first).float().permute(2, 0, 1) / 255.0
            
            # [2, 3, H, W]
            first_frame = torch.stack([main_first_t, wrist_first_t], dim=0)

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

        return {
            "video": video,             # Shape: [2, 3, 16, 224, 224]
            "state": state_input,       # Shape: [16, 8]
            "action_target": action_target, # Shape: [16, 8] (Sequence)
            "text_tokens": text_tokens,
            "first_frame": first_frame, # Shape: [2, 3, 224, 224]
            "teacher_siglip": teacher_siglip,
            "teacher_exo": teacher_exo
        }