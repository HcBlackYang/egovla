# # utils/dataset_loader.py
# import torch
# from torch.utils.data import Dataset
# import h5py
# import numpy as np
# import os
# import json
# from transformers import T5Tokenizer

# class RobotDataset(Dataset):
#     def __init__(self, hdf5_path, 
#                  window_size=16, 
#                  pred_horizon=64,
#                  tokenizer_path="/yanghaochuan/models/flan-t5-large",
#                  stats_path="/yanghaochuan/data/16dataset_stats.json"): 
        
#         self.hdf5_path = hdf5_path
#         self.window_size = window_size
#         self.pred_horizon = pred_horizon
        
#         # === 1. 加载 Tokenizer ===
#         print(f"[Dataset] Loading Tokenizer from {tokenizer_path}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
#         except:
#             print("[Dataset] Local tokenizer failed, trying default...")
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
#         # === 2. 加载统计量 (用于 Z-Score 归一化) ===
#         if not os.path.exists(stats_path):
#              raise FileNotFoundError(f"❌ 找不到统计文件: {stats_path}。请先运行 utils/compute_stats.py！")
        
#         with open(stats_path, 'r') as f:
#             stats = json.load(f)
        
#         self.action_mean = torch.tensor(stats['action_mean']).float()
#         self.action_std = torch.tensor(stats['action_std']).float()
#         self.action_std = torch.maximum(self.action_std, torch.tensor(1e-2))
        
#         # === 3. 扫描数据并建立 Anchor 缓存 (Index-Based Asymmetric Context) ===
#         self.indices = []
#         self.anchor_bank = {}  # {demo_key: first_frame_tensor}
        
#         print(f"[Dataset] Scanning HDF5 for valid samples and Anchors...")
        
#         with h5py.File(hdf5_path, 'r') as f:
#             if 'data' not in f:
#                  raise ValueError(f"HDF5结构错误: {hdf5_path} 中没有 'data' 组")

#             self.demos = list(f['data'].keys())
            
#             # --- 第一遍扫描：收集所有 Type B (Anchors) ---
#             for demo_key in self.demos:
#                 demo_grp = f['data'][demo_key]
                
#                 # 优先读取 HDF5 中的属性标记
#                 # 如果是旧数据没有标记，回退到 demo_idx % 5 == 0 的逻辑
#                 data_type = demo_grp.attrs.get("data_type", None)
#                 if data_type is None:
#                     idx = int(demo_key.split('_')[1])
#                     if idx % 5 == 0:
#                         data_type = "type_b"
                
#                 # 如果被标记为 Type B，则存入 Anchor 银行
#                 if data_type == "type_b":
#                     main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
#                     wrist_key = 'robot0_eye_in_hand_image'
                    
#                     if main_key in demo_grp['obs'] and wrist_key in demo_grp['obs']:
#                         m0 = torch.tensor(demo_grp['obs'][main_key][0]).float().permute(2, 0, 1) / 255.0
#                         w0 = torch.tensor(demo_grp['obs'][wrist_key][0]).float().permute(2, 0, 1) / 255.0
                        
#                         # [2, 3, H, W]
#                         anchor_frame = torch.stack([m0, w0], dim=0)
#                         self.anchor_bank[demo_key] = anchor_frame
            
#             print(f"[Dataset] Identified {len(self.anchor_bank)} anchors (Type B episodes).")

#             # --- 第二遍扫描：构建训练样本索引 ---
#             for demo_key in self.demos:
#                 try:
#                     demo_grp = f['data'][demo_key]
#                     if 'actions' not in demo_grp: continue
                    
#                     total_len = demo_grp['actions'].shape[0]
#                     has_teacher = 'teacher_siglip' in demo_grp
#                     min_len = window_size + pred_horizon
                    
#                     if total_len > min_len:
#                         instruction = demo_grp.attrs.get('language_instruction', 'do nothing')
#                         if isinstance(instruction, bytes): instruction = instruction.decode('utf-8')

#                         for i in range(total_len - min_len): 
#                             self.indices.append({
#                                 'demo_key': demo_key,
#                                 'start_idx': i,
#                                 'instruction': instruction,
#                                 'has_teacher': has_teacher
#                             })
#                 except Exception as e:
#                     print(f"Skipping {demo_key}: {e}")
        
#         print(f"[Dataset] Loaded {len(self.indices)} samples.")

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         meta = self.indices[idx]
#         demo_key = meta['demo_key']
#         start = meta['start_idx']
#         instruction = meta['instruction']
        
#         read_len = self.window_size + self.pred_horizon
        
#         with h5py.File(self.hdf5_path, 'r') as f:
#             demo_grp = f['data'][demo_key]
            
#             # --- 1. Video (读取 Type A 的真实“糟糕”视野) ---
#             main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
#             wrist_key = 'robot0_eye_in_hand_image'
            
#             main_seq = demo_grp['obs'][main_key][start : start + self.window_size]
#             wrist_seq = demo_grp['obs'][wrist_key][start : start + self.window_size]
            
#             main_t = torch.tensor(main_seq).float().permute(0, 3, 1, 2) / 255.0
#             wrist_t = torch.tensor(wrist_seq).float().permute(0, 3, 1, 2) / 255.0
            
#             # [2, 16, 3, H, W] -> [2, 3, 16, H, W]
#             video = torch.stack([main_t, wrist_t], dim=0).permute(0, 2, 1, 3, 4)
            
#             # --- 2. State & Action ---
#             state_seq_raw = demo_grp['obs']['robot0_joint_pos'][start : start + read_len]
#             if state_seq_raw.shape[0] < read_len:
#                 pad_len = read_len - state_seq_raw.shape[0]
#                 state_seq_raw = np.concatenate([state_seq_raw, np.tile(state_seq_raw[-1:], (pad_len, 1))], axis=0)

#             state_seq_tensor = torch.tensor(state_seq_raw).float()
#             state_seq_norm = (state_seq_tensor - self.action_mean) / self.action_std
            
#             state_input = state_seq_norm[:self.window_size]
#             action_target = state_seq_norm[self.window_size : self.window_size + self.pred_horizon]

#             # --- 3. First Frame (Context Injection) ---
#             # 关键修改：按索引分组查找 Anchor
#             # 解析当前索引: "demo_12" -> 12
#             current_idx = int(demo_key.split('_')[1])
            
#             # 计算归属的 Anchor 索引 (向下取整到最近的 5 的倍数)
#             # 例如: 12 -> 10,  14 -> 10,  15 -> 15
#             anchor_idx = (current_idx // 5) * 5
#             anchor_key = f"demo_{anchor_idx}"
            
#             if anchor_key in self.anchor_bank:
#                 # 命中缓存：使用对应的 Type B 首帧
#                 first_frame = self.anchor_bank[anchor_key]
#             else:
#                 # Fallback: 理论上不应该发生，除非 Type B 被过滤了
#                 # 如果找不到，就用自己的首帧
#                 m0 = torch.tensor(demo_grp['obs'][main_key][0]).float().permute(2, 0, 1) / 255.0
#                 w0 = torch.tensor(demo_grp['obs'][wrist_key][0]).float().permute(2, 0, 1) / 255.0
#                 first_frame = torch.stack([m0, w0], dim=0)

#             # --- 4. Teachers ---
#             if meta['has_teacher']:
#                 teacher_siglip = torch.tensor(demo_grp['teacher_siglip'][start : start + self.window_size]).float()
#                 teacher_exo = torch.tensor(demo_grp['teacher_exo'][start : start + self.window_size]).float()
#             else:
#                 teacher_siglip = torch.zeros(self.window_size, 1152)
#                 teacher_exo = torch.zeros(self.window_size, 1152)

#         # Tokenize
#         text_tokens = self.tokenizer(
#             instruction, return_tensors="pt", padding="max_length", max_length=16, truncation=True
#         ).input_ids.squeeze(0)

#         return {
#             "video": video,
#             "state": state_input,
#             "action_target": action_target,
#             "text_tokens": text_tokens,
#             "first_frame": first_frame, # <--- Swapped Context (Type B)
#             "teacher_siglip": teacher_siglip,
#             "teacher_exo": teacher_exo
#         }
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import json
from transformers import T5Tokenizer

class RobotDataset(Dataset):
    def __init__(self, hdf5_path, 
                 window_size=6,         # 🟢 修改：实际输入给模型的帧数 (从16改为6)
                 history_len=500,        # 🟢 新增：模拟的历史视野长度 (从中采样6帧)
                 pred_horizon=64,
                 tokenizer_path="/yanghaochuan/models/flan-t5-large",
                 stats_path="/yanghaochuan/data/111dataset_stats.json"): 
        
        self.hdf5_path = hdf5_path
        self.window_size = window_size   # 输出给模型的帧数 (6)
        self.history_len = history_len   # 历史采样窗口 (48)
        self.pred_horizon = pred_horizon
        
        # 🟢 定义稀疏预测步长 (World Model Anchors)
        self.future_offsets = [0, 2, 4, 8, 16, 32]
        
        # === 1. 加载 Tokenizer ===
        print(f"[Dataset] Loading Tokenizer from {tokenizer_path}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except:
            print("[Dataset] Local tokenizer failed, trying default...")
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
        # === 2. 加载统计量 ===
        if not os.path.exists(stats_path):
             raise FileNotFoundError(f"❌ 找不到统计文件: {stats_path}")
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.action_mean = torch.tensor(stats['action_mean']).float()
        self.action_std = torch.tensor(stats['action_std']).float()
        self.action_std = torch.maximum(self.action_std, torch.tensor(1e-2))
        
        # === 3. 扫描数据 ===
        self.indices = []
        self.anchor_bank = {}
        
        print(f"[Dataset] Scanning HDF5...")
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f: raise ValueError(f"HDF5结构错误")
            self.demos = list(f['data'].keys())
            
            # --- 收集 Anchors (Type B) ---
            for demo_key in self.demos:
                demo_grp = f['data'][demo_key]
                # 兼容旧数据的 Type B 判定
                data_type = demo_grp.attrs.get("data_type", None)
                if data_type is None: 
                    idx = int(demo_key.split('_')[1])
                    if idx % 5 == 0: data_type = "type_b"
                
                if data_type == "type_b":
                    main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
                    wrist_key = 'robot0_eye_in_hand_image'
                    if main_key in demo_grp['obs']:
                        m0 = torch.tensor(demo_grp['obs'][main_key][0]).float().permute(2, 0, 1) / 255.0
                        w0 = torch.tensor(demo_grp['obs'][wrist_key][0]).float().permute(2, 0, 1) / 255.0
                        self.anchor_bank[demo_key] = torch.stack([m0, w0], dim=0)

            # --- 构建样本索引 ---
            # 注意：这里的 start_idx 代表的是“当前时刻 t”的基准点
            # 实际上，我们需要确保 t + pred_horizon 不越界
            # 历史数据不够 history_len 时，我们会用首帧填充 (Handling Cold Start)
            for demo_key in self.demos:
                demo_grp = f['data'][demo_key]
                if 'actions' not in demo_grp: continue
                total_len = demo_grp['actions'].shape[0]
                
                # 只要剩余长度够预测未来即可
                if total_len > self.pred_horizon:
                    instr = demo_grp.attrs.get('language_instruction', 'do nothing')
                    if isinstance(instr, bytes): instr = instr.decode('utf-8')
                    has_teacher = 'teacher_siglip' in demo_grp
                    
                    # 我们让 i 代表 "当前时刻 t"
                    # 遍历范围：从 0 到 total_len - pred_horizon
                    for i in range(total_len - self.pred_horizon): 
                        self.indices.append({
                            'demo_key': demo_key, 
                            'current_t': i, 
                            'instruction': instr, 
                            'has_teacher': has_teacher
                        })
        print(f"[Dataset] Loaded {len(self.indices)} samples.")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        meta = self.indices[idx]
        demo_key = meta['demo_key']
        current_t = meta['current_t'] # 当前时刻 t
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_grp = f['data'][demo_key]
            demo_len = demo_grp['actions'].shape[0]

            # === 1. Video: 动态均匀采样 (Uniform Sampling) ===
            main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
            wrist_key = 'robot0_eye_in_hand_image'
            
            # 确定历史窗口: [t - history_len + 1, t]
            # 例如 t=10, len=48 -> start=-37 (越界) -> 实际上只有 0~10 可用
            history_start = max(0, current_t - self.history_len + 1)
            history_end = current_t + 1 # 切片是不包含end的，所以+1以包含t
            
            valid_len = history_end - history_start
            
            # 计算均匀采样索引 (在 valid_len 范围内选 window_size 帧)
            # 例如从 100 帧里选 6 帧 -> [0, 19, 39, ..., 99]
            if valid_len < self.window_size:
                # 冷启动策略：如果历史不够长 (例如刚开始第2帧)，怎么选6帧？
                # 策略：重复利用现有帧，或者全部取完后用首帧填充。
                # np.linspace 在 valid_len < num 时会自动处理 (产生重复索引，如 [0,0,1,1,2,2])
                # 这正是我们想要的 "Copy First Frame" 的泛化版本
                offsets = np.linspace(0, valid_len - 1, self.window_size).astype(int)
            else:
                offsets = np.linspace(0, valid_len - 1, self.window_size).astype(int)
            
            # 映射回全局索引
            global_indices = history_start + offsets
            # 排序确保时序正确 (linspace 已经是递增的，保险起见)
            global_indices = np.sort(global_indices)
            
            # 读取视频 (HDF5 支持列表索引)
            # [6, H, W, 3]
            # main_frames = demo_grp['obs'][main_key][global_indices]
            # wrist_frames = demo_grp['obs'][wrist_key][global_indices]

            # 🟢 [修复开始]：h5py 不支持重复索引，必须先去重再映射
            # 1. 获取唯一索引和重建映射表
            unique_indices, inverse_indices = np.unique(global_indices, return_inverse=True)
            
            # 2. 只读取唯一的帧 (h5py 要求严格递增，unique 自动排好序了)
            # 读出来是 [U, H, W, 3]，其中 U <= window_size
            unique_main_frames = demo_grp['obs'][main_key][unique_indices]
            unique_wrist_frames = demo_grp['obs'][wrist_key][unique_indices]
            
            # 3. 在内存中重建完整序列 (包含重复帧)
            # 使用 inverse_indices 把 [U, ...] 映射回 [6, ...]
            main_frames = unique_main_frames[inverse_indices]
            wrist_frames = unique_wrist_frames[inverse_indices]
            
            # 转 Tensor [6, 3, H, W]
            main_seq = torch.tensor(main_frames).float().permute(0, 3, 1, 2) / 255.0
            wrist_seq = torch.tensor(wrist_frames).float().permute(0, 3, 1, 2) / 255.0
            
            # Stack Views: [2, 3, 6, H, W]
            video = torch.stack([main_seq, wrist_seq], dim=0).permute(0, 1, 2, 3, 4) # 这里的 dim 顺序按你模型要求来
            # 注意：之前是 [2, 3, T, H, W] 还是 [B, 2, C, T, H, W]?
            # 你的旧代码是: torch.stack([main_tensor, wrist_tensor], dim=0).permute(0, 2, 1, 3, 4)
            # 即 [2, T, 3, H, W] -> [2, 3, T, H, W]
            # 这里 main_seq 是 [T, 3, H, W]，所以 permute 后是 [2, 3, T, H, W]
            video = torch.stack([main_seq, wrist_seq], dim=0).transpose(1, 2) 

            # === 2. State & Action (RDT 仍然需要未来的 Action) ===
            # State: 取当前时刻 t 的状态 (作为 Condition)
            # Action: 取 t 到 t + pred_horizon
            state_raw = demo_grp['obs']['robot0_joint_pos'][current_t : current_t + self.pred_horizon + 1]
            
            # 补齐
            target_len = self.pred_horizon + 1 # 1个当前State + K个Action
            if state_raw.shape[0] < target_len:
                state_raw = np.concatenate([state_raw, np.tile(state_raw[-1:], (target_len-state_raw.shape[0], 1))], axis=0)
            
            state_norm = (torch.tensor(state_raw).float() - self.action_mean) / self.action_std
            
            state_input = state_norm[:1] # [1, 8] - 当前 State
            # 如果 RDT 需要历史 State 序列，这里要改。但根据你的ForeSight设计，RDT用当前State+Latent即可。
            # 为了兼容你之前的 dataset (返回 window_size 个 state)，我们可以填充
            # 但新逻辑下，State 主要是当前状态。
            # 这里为了兼容性，返回 [16, 8]，前面用当前状态填充
            state_input_expanded = state_norm[0].unsqueeze(0).repeat(self.window_size, 1) # [6, 8]
            
            action_target = state_norm[1:] # [64, 8]

            # === 3. First Frame (Anchor) ===
            curr_idx = int(demo_key.split('_')[1])
            anchor_key = f"demo_{(curr_idx//5)*5}"
            first_frame = self.anchor_bank.get(anchor_key, video[:, :, 0]) # Fallback to current start

            # === 4. Teachers (World Model Targets: Sparse Future) ===
            future_exo_feats = []
            if meta['has_teacher']:
                # 读取 t, t+4, t+8... 的特征
                for offset in self.future_offsets:
                    target_idx = min(current_t + offset, demo_len - 1)
                    future_exo_feats.append(torch.from_numpy(demo_grp['teacher_exo'][target_idx]).float())
                future_exo_target = torch.stack(future_exo_feats)
                
                # 语义辅助 (取当前窗口的平均，或者直接取当前帧语义)
                # 为了简单，取当前 t 的 SigLIP
                teacher_siglip = torch.from_numpy(demo_grp['teacher_siglip'][current_t]).float().unsqueeze(0).repeat(self.window_size, 1)
            else:
                teacher_siglip = torch.zeros(self.window_size, 1152)
                future_exo_target = torch.zeros(len(self.future_offsets), 1152)
                
            # Teacher Exo Legacy (为了兼容旧接口，全0即可，或者取当前的)
            teacher_exo_legacy = torch.zeros(self.window_size, 1152)

        text_tokens = self.tokenizer(meta['instruction'], return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids.squeeze(0)

        return {
            "video": video,                 # [2, 3, 6, H, W] (Uniform Sampled)
            "state": state_input_expanded,  # [6, 8] (Current State repeated)
            "action_target": action_target, # [64, 8]
            "text_tokens": text_tokens,
            "first_frame": first_frame,
            "teacher_siglip": teacher_siglip,
            "teacher_exo": teacher_exo_legacy,
            "future_exo_target": future_exo_target # [6, 1152] (Sparse Future)
        }