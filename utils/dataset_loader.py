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



# import torch
# from torch.utils.data import Dataset
# import h5py
# import numpy as np
# import os
# import json
# from transformers import T5Tokenizer
# from torchvision import transforms

# class RobotDataset(Dataset):
#     def __init__(self, hdf5_path, in_memory=True, 
#                  window_size=6,         # 🟢 修改：实际输入给模型的帧数 (从16改为6)
#                  history_len=500,        # 🟢 新增：模拟的历史视野长度 (从中采样6帧)
#                  pred_horizon=64,
#                  tokenizer_path="/yanghaochuan/models/flan-t5-large",
#                  stats_path="/yanghaochuan/data/111dataset_stats.json"): 
        
#         self.hdf5_path = hdf5_path
#         self.window_size = window_size   # 输出给模型的帧数 (6)
#         self.history_len = history_len   # 历史采样窗口 (48)
#         self.pred_horizon = pred_horizon
        
#         # 🟢 定义稀疏预测步长 (World Model Anchors)
#         self.future_offsets = [0, 2, 4, 8, 16, 32]

#         # === [新增] 定义归一化 (VideoMAE 标准) ===
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                               std=[0.229, 0.224, 0.225])
#         # === 1. 加载 Tokenizer ===
#         print(f"[Dataset] Loading Tokenizer from {tokenizer_path}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
#         except:
#             print("[Dataset] Local tokenizer failed, trying default...")
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
#         # === 2. 加载统计量 ===
#         if not os.path.exists(stats_path):
#              raise FileNotFoundError(f"❌ 找不到统计文件: {stats_path}")
        
#         with open(stats_path, 'r') as f:
#             stats = json.load(f)
        
#         self.action_mean = torch.tensor(stats['action_mean']).float()
#         self.action_std = torch.tensor(stats['action_std']).float()
#         self.action_std = torch.maximum(self.action_std, torch.tensor(1e-2))
        
#         # === 3. 扫描数据 ===
#         self.indices = []
#         self.anchor_bank = {}
        
#         print(f"[Dataset] Scanning HDF5...")
#         with h5py.File(hdf5_path, 'r') as f:
#             if 'data' not in f: raise ValueError(f"HDF5结构错误")
#             self.demos = list(f['data'].keys())
            
#             # --- 收集 Anchors (Type B) ---
#             for demo_key in self.demos:
#                 demo_grp = f['data'][demo_key]
#                 # 兼容旧数据的 Type B 判定
#                 data_type = demo_grp.attrs.get("data_type", None)
#                 if data_type is None: 
#                     idx = int(demo_key.split('_')[1])
#                     if idx % 5 == 0: data_type = "type_b"
                
#                 if data_type == "type_b":
#                     main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
#                     wrist_key = 'robot0_eye_in_hand_image'
#                     if main_key in demo_grp['obs']:
#                         m0 = torch.tensor(demo_grp['obs'][main_key][0]).float().permute(2, 0, 1) / 255.0
#                         w0 = torch.tensor(demo_grp['obs'][wrist_key][0]).float().permute(2, 0, 1) / 255.0
#                         self.anchor_bank[demo_key] = torch.stack([m0, w0], dim=0)

#             # --- 构建样本索引 ---
#             # 注意：这里的 start_idx 代表的是“当前时刻 t”的基准点
#             # 实际上，我们需要确保 t + pred_horizon 不越界
#             # 历史数据不够 history_len 时，我们会用首帧填充 (Handling Cold Start)
#             for demo_key in self.demos:
#                 demo_grp = f['data'][demo_key]
#                 if 'actions' not in demo_grp: continue
#                 total_len = demo_grp['actions'].shape[0]
                
#                 # 只要剩余长度够预测未来即可
#                 if total_len > self.pred_horizon:
#                     instr = demo_grp.attrs.get('language_instruction', 'do nothing')
#                     if isinstance(instr, bytes): instr = instr.decode('utf-8')
#                     has_teacher = 'teacher_siglip' in demo_grp
                    
#                     # 我们让 i 代表 "当前时刻 t"
#                     # 遍历范围：从 0 到 total_len - pred_horizon
#                     for i in range(total_len - self.pred_horizon): 
#                         self.indices.append({
#                             'demo_key': demo_key, 
#                             'current_t': i, 
#                             'instruction': instr, 
#                             'has_teacher': has_teacher
#                         })
#         print(f"[Dataset] Loaded {len(self.indices)} samples.")

#     def __len__(self): return len(self.indices)

#     def __getitem__(self, idx):
#         meta = self.indices[idx]
#         demo_key = meta['demo_key']
#         current_t = meta['current_t'] # 当前时刻 t
        
#         with h5py.File(self.hdf5_path, 'r') as f:
#             demo_grp = f['data'][demo_key]
#             demo_len = demo_grp['actions'].shape[0]

#             # === 1. Video: 动态均匀采样 (Uniform Sampling) ===
#             main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
#             wrist_key = 'robot0_eye_in_hand_image'
            
#             # 确定历史窗口: [t - history_len + 1, t]
#             # 例如 t=10, len=48 -> start=-37 (越界) -> 实际上只有 0~10 可用
#             history_start = max(0, current_t - self.history_len + 1)
#             history_end = current_t + 1 # 切片是不包含end的，所以+1以包含t
            
#             valid_len = history_end - history_start
            
#             # 计算均匀采样索引 (在 valid_len 范围内选 window_size 帧)
#             # 例如从 100 帧里选 6 帧 -> [0, 19, 39, ..., 99]
#             if valid_len < self.window_size:
#                 # 冷启动策略：如果历史不够长 (例如刚开始第2帧)，怎么选6帧？
#                 # 策略：重复利用现有帧，或者全部取完后用首帧填充。
#                 # np.linspace 在 valid_len < num 时会自动处理 (产生重复索引，如 [0,0,1,1,2,2])
#                 # 这正是我们想要的 "Copy First Frame" 的泛化版本
#                 offsets = np.linspace(0, valid_len - 1, self.window_size).astype(int)
#             else:
#                 offsets = np.linspace(0, valid_len - 1, self.window_size).astype(int)
            
#             # 映射回全局索引
#             global_indices = history_start + offsets
#             # 排序确保时序正确 (linspace 已经是递增的，保险起见)
#             global_indices = np.sort(global_indices)
            
#             # 读取视频 (HDF5 支持列表索引)
#             # [6, H, W, 3]
#             # main_frames = demo_grp['obs'][main_key][global_indices]
#             # wrist_frames = demo_grp['obs'][wrist_key][global_indices]

#             # 🟢 [修复开始]：h5py 不支持重复索引，必须先去重再映射
#             # 1. 获取唯一索引和重建映射表
#             unique_indices, inverse_indices = np.unique(global_indices, return_inverse=True)
            
#             # 2. 只读取唯一的帧 (h5py 要求严格递增，unique 自动排好序了)
#             # 读出来是 [U, H, W, 3]，其中 U <= window_size
#             unique_main_frames = demo_grp['obs'][main_key][unique_indices]
#             unique_wrist_frames = demo_grp['obs'][wrist_key][unique_indices]
            
#             # 3. 在内存中重建完整序列 (包含重复帧)
#             # 使用 inverse_indices 把 [U, ...] 映射回 [6, ...]
#             main_frames = unique_main_frames[inverse_indices]
#             wrist_frames = unique_wrist_frames[inverse_indices]
            
#             # # 转 Tensor [6, 3, H, W]
#             # main_seq = torch.tensor(main_frames).float().permute(0, 3, 1, 2) / 255.0
#             # wrist_seq = torch.tensor(wrist_frames).float().permute(0, 3, 1, 2) / 255.0
            
#             # # Stack Views: [2, 3, 6, H, W]
#             # video = torch.stack([main_seq, wrist_seq], dim=0).permute(0, 1, 2, 3, 4) # 这里的 dim 顺序按你模型要求来

#             # 修改后 (先 /255.0，再归一化):
#             main_t_raw = torch.tensor(main_frames).float().permute(0, 3, 1, 2) / 255.0
#             wrist_t_raw = torch.tensor(wrist_frames).float().permute(0, 3, 1, 2) / 255.0
            
#             # 应用归一化 (注意维度匹配，Normalize作用于C维度)
#             # main_t_raw shape: [6, 3, H, W]
#             main_seq = self.normalize(main_t_raw)
#             wrist_seq = self.normalize(wrist_t_raw)
            
#             # Stack Views: [2, 3, 6, H, W] (根据你的模型要求调整)
#             video = torch.stack([main_seq, wrist_seq], dim=0).transpose(1, 2)


#             # 注意：之前是 [2, 3, T, H, W] 还是 [B, 2, C, T, H, W]?
#             # 你的旧代码是: torch.stack([main_tensor, wrist_tensor], dim=0).permute(0, 2, 1, 3, 4)
#             # 即 [2, T, 3, H, W] -> [2, 3, T, H, W]
#             # 这里 main_seq 是 [T, 3, H, W]，所以 permute 后是 [2, 3, T, H, W]
#             video = torch.stack([main_seq, wrist_seq], dim=0).transpose(1, 2) 

#             # === 2. State & Action (RDT 仍然需要未来的 Action) ===
#             # State: 取当前时刻 t 的状态 (作为 Condition)
#             # Action: 取 t 到 t + pred_horizon
#             state_raw = demo_grp['obs']['robot0_joint_pos'][current_t : current_t + self.pred_horizon + 1]
            
#             # 补齐
#             target_len = self.pred_horizon + 1 # 1个当前State + K个Action
#             if state_raw.shape[0] < target_len:
#                 state_raw = np.concatenate([state_raw, np.tile(state_raw[-1:], (target_len-state_raw.shape[0], 1))], axis=0)
            
#             state_norm = (torch.tensor(state_raw).float() - self.action_mean) / self.action_std
            
#             state_input = state_norm[:1] # [1, 8] - 当前 State
#             # 如果 RDT 需要历史 State 序列，这里要改。但根据你的ForeSight设计，RDT用当前State+Latent即可。
#             # 为了兼容你之前的 dataset (返回 window_size 个 state)，我们可以填充
#             # 但新逻辑下，State 主要是当前状态。
#             # 这里为了兼容性，返回 [16, 8]，前面用当前状态填充
#             state_input_expanded = state_norm[0].unsqueeze(0).repeat(self.window_size, 1) # [6, 8]
            
#             action_target = state_norm[1:] # [64, 8]

#             # === 3. First Frame (Anchor) ===
#             curr_idx = int(demo_key.split('_')[1])
#             anchor_key = f"demo_{(curr_idx//5)*5}"
#             first_frame = self.anchor_bank.get(anchor_key, video[:, :, 0]) # Fallback to current start

#             # === 4. Teachers (World Model Targets: Sparse Future) ===
#             future_exo_feats = []
#             if meta['has_teacher']:
#                 # 读取 t, t+4, t+8... 的特征
#                 for offset in self.future_offsets:
#                     target_idx = min(current_t + offset, demo_len - 1)
#                     future_exo_feats.append(torch.from_numpy(demo_grp['teacher_exo'][target_idx]).float())
#                 future_exo_target = torch.stack(future_exo_feats)
                
#                 # 语义辅助 (取当前窗口的平均，或者直接取当前帧语义)
#                 # 为了简单，取当前 t 的 SigLIP
#                 teacher_siglip = torch.from_numpy(demo_grp['teacher_siglip'][current_t]).float().unsqueeze(0).repeat(self.window_size, 1)
#             else:
#                 teacher_siglip = torch.zeros(self.window_size, 1152)
#                 future_exo_target = torch.zeros(len(self.future_offsets), 1152)
                
#             # Teacher Exo Legacy (为了兼容旧接口，全0即可，或者取当前的)
#             teacher_exo_legacy = torch.zeros(self.window_size, 1152)

#         text_tokens = self.tokenizer(meta['instruction'], return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids.squeeze(0)

#         if anchor_key in self.anchor_bank:
#             first_frame = self.anchor_bank[anchor_key] 
#             # 注意：如果是从 bank 里取出的，确保 bank 存的时候也归一化了，或者在这里归一化
#             # 如果 bank 里存的是 raw tensor，这里要加:
#             # first_frame = self.normalize(first_frame) 
#         else:
#             # 如果是现场读取
#             m0 = torch.tensor(demo_grp['obs'][main_key][0]).float().permute(2, 0, 1) / 255.0
#             w0 = torch.tensor(demo_grp['obs'][wrist_key][0]).float().permute(2, 0, 1) / 255.0
            
#             # 也要归一化
#             m0 = self.normalize(m0)
#             w0 = self.normalize(w0)
#             first_frame = torch.stack([m0, w0], dim=0)


#         return {
#             "video": video,                 # [2, 3, 6, H, W] (Uniform Sampled)
#             "state": state_input_expanded,  # [6, 8] (Current State repeated)
#             "action_target": action_target, # [64, 8]
#             "text_tokens": text_tokens,
#             "first_frame": first_frame,
#             "teacher_siglip": teacher_siglip,
#             "teacher_exo": teacher_exo_legacy,
#             "future_exo_target": future_exo_target # [6, 1152] (Sparse Future)
#         }

# utils/dataset_loader.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
import json
from transformers import T5Tokenizer
from torchvision import transforms
from tqdm import tqdm

class RobotDataset(Dataset):
    def __init__(self, hdf5_path, in_memory=True, 
                 window_size=6,         # 实际输入给模型的帧数
                 history_len=500,       # 模拟的历史视野长度 (从中采样 window_size 帧)
                 pred_horizon=64,
                 tokenizer_path="/yanghaochuan/models/flan-t5-large",
                 stats_path="/yanghaochuan/data/130dataset_stats.json"): 
        
        self.hdf5_path = hdf5_path
        self.window_size = window_size
        self.history_len = history_len
        self.pred_horizon = pred_horizon
        self.in_memory = in_memory
        
        # 🟢 定义稀疏预测步长 (World Model Anchors)
        self.future_offsets = [0, 2, 4, 8, 16, 32]

        # === 定义归一化 (VideoMAE 标准) ===
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

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
        
        # === 3. 扫描数据结构 & Anchor ===
        self.indices = []
        self.anchor_bank = {}
        self.cache = {} # 内存缓存

        print(f"[Dataset] Scanning HDF5 structure...")
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f: raise ValueError(f"HDF5结构错误")
            self.demos = list(f['data'].keys())
            
            # --- 3.1 收集 Anchors (Type B) ---
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
                        # 存入 Anchor Bank (注意：这里暂未归一化，使用时再处理，或者在此处统一)
                        # 为了统一，我们在 __getitem__ 处统一做 Normalize
                        self.anchor_bank[demo_key] = torch.stack([m0, w0], dim=0)

            # --- 3.2 暴力预加载到内存 (IO Boost) ---
            if self.in_memory:
                print(f"📥 [IO Boost] Loading ALL data to RAM (Total: {len(self.demos)} demos)...")
                for demo_key in tqdm(self.demos):
                    grp = f['data'][demo_key]
                    cache_item = {}
                    
                    # 自动识别 Main Camera Key
                    main_key = 'agentview_image' if 'agentview_image' in grp['obs'] else 'agentview_rgb'
                    wrist_key = 'robot0_eye_in_hand_image'

                    # 读取图像数据 (耗时操作)
                    cache_item['main_img'] = grp['obs'][main_key][:]
                    cache_item['wrist_img'] = grp['obs'][wrist_key][:]
                    
                    # 读取状态与动作
                    cache_item['qpos'] = grp['obs']['robot0_joint_pos'][:]
                    # 如果有显式的 actions dataset 就读，没有则后续通过 qpos 切片
                    if 'actions' in grp:
                        cache_item['actions'] = grp['actions'][:]
                    
                    # 读取 Teacher 特征
                    if 'teacher_siglip' in grp:
                        cache_item['teacher_siglip'] = grp['teacher_siglip'][:]
                    if 'teacher_exo' in grp:
                        cache_item['teacher_exo'] = grp['teacher_exo'][:]
                    
                    self.cache[demo_key] = cache_item
                print("✅ Dataset successfully loaded to RAM!")

            # --- 3.3 构建样本索引 ---
            for demo_key in self.demos:
                # 如果在内存里，直接查缓存；否则查文件
                if self.in_memory:
                    # 使用缓存中的长度信息
                    if 'actions' in self.cache[demo_key]:
                        total_len = self.cache[demo_key]['actions'].shape[0]
                    else:
                        total_len = self.cache[demo_key]['qpos'].shape[0] # Fallback
                else:
                    demo_grp = f['data'][demo_key]
                    if 'actions' not in demo_grp: continue
                    total_len = demo_grp['actions'].shape[0]

                # 只要剩余长度够预测未来即可
                if total_len > self.pred_horizon:
                    # 获取指令 (指令通常很短，暂不缓存，每次读取开销可忽略，或者存入 meta)
                    # 为简单起见，这里还是每次读 attributes，或者如果你想极致优化，也可以缓存指令
                    if self.in_memory:
                        # HDF5 attributes 无法离线缓存，这里我们偷个懒，只缓存数据
                        # 实际运行时 Instruction 读取很快
                        pass 
                    
                    # 为了获取 instruction，如果是 disk mode 必须读 file
                    # 如果是 memory mode，我们这里依然需要 file handle 来读 attrs
                    # 但 dataset scan 只跑一次，所以没关系
                    demo_grp = f['data'][demo_key]
                    instr = demo_grp.attrs.get('language_instruction', 'do nothing')
                    if isinstance(instr, bytes): instr = instr.decode('utf-8')
                    
                    has_teacher = False
                    if self.in_memory:
                        has_teacher = 'teacher_siglip' in self.cache[demo_key]
                    else:
                        has_teacher = 'teacher_siglip' in demo_grp
                    
                    # # 遍历每一个时间步
                    # for i in range(total_len - self.pred_horizon): 
                    #     self.indices.append({
                    #         'demo_key': demo_key, 
                    #         'current_t': i, 
                    #         'instruction': instr, 
                    #         'has_teacher': has_teacher
                    #     })
                    # 1. 判定当前 Demo 是否为 Type B (初始位置固定的数据)
                    # 假设你的命名规则是 demo_0, demo_5, demo_10... 是 Type B
                    curr_idx = int(demo_key.split('_')[1])
                    is_type_b = (curr_idx % 5 == 0) 

                    # 2. 设置重复次数
                    # Type B (20条) 重复 4 次 -> 等效 80 条
                    # Type A (80条) 重复 1 次 -> 等效 80 条
                    # 这样总比例接近 1:1
                    repeat_times = 4 if is_type_b else 1

                    for _ in range(repeat_times):  # <--- 🟢 新增循环：实现过采样
                        for i in range(total_len - self.pred_horizon): 
                            self.indices.append({
                                'demo_key': demo_key, 
                                'current_t': i, 
                                'instruction': instr, 
                                'has_teacher': has_teacher
                            })

                    print(f"[Dataset Loader] Demo {demo_key} (Type B={is_type_b}) loaded {repeat_times} times.")
                        
        print(f"[Dataset] Loaded {len(self.indices)} samples.")

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        meta = self.indices[idx]
        demo_key = meta['demo_key']
        current_t = meta['current_t']
        
        # 准备数据容器
        main_frames = None
        wrist_frames = None
        state_raw = None
        teacher_siglip_tensor = None
        future_exo_target = None
        
        # 确定历史窗口
        history_start = max(0, current_t - self.history_len + 1)
        history_end = current_t + 1
        valid_len = history_end - history_start
        
        # 计算采样索引
        offsets = np.linspace(0, valid_len - 1, self.window_size).astype(int)
        global_indices = history_start + offsets
        global_indices = np.sort(global_indices)

        # =========================================================
        # 🟢 分支 A: 内存极速读取 (In-Memory)
        # =========================================================
        if self.in_memory:
            demo_data = self.cache[demo_key]
            demo_len = demo_data['qpos'].shape[0] # 使用 qpos 长度作为基准
            
            # Numpy 支持重复索引，直接读取即可
            main_frames = demo_data['main_img'][global_indices]
            wrist_frames = demo_data['wrist_img'][global_indices]
            
            # State (从 current_t 开始)
            # RDT 需要: State(t) + Action(t...t+H)
            # 这里统一从 qpos 读取
            state_range_end = min(current_t + self.pred_horizon + 1, demo_len)
            state_raw = demo_data['qpos'][current_t : state_range_end]

            # Teacher
            if meta['has_teacher']:
                # SigLIP: 取当前帧
                teacher_siglip_tensor = torch.from_numpy(demo_data['teacher_siglip'][current_t]).float()
                
                # Future Exo: 稀疏采样
                feats = []
                for offset in self.future_offsets:
                    target_idx = min(current_t + offset, demo_len - 1)
                    feats.append(torch.from_numpy(demo_data['teacher_exo'][target_idx]).float())
                future_exo_target = torch.stack(feats)


        
        # =========================================================
        # 🔵 分支 B: 硬盘读取 (Disk - 兼容旧逻辑)
        # =========================================================
        else:
            with h5py.File(self.hdf5_path, 'r') as f:
                demo_grp = f['data'][demo_key]
                demo_len = demo_grp['actions'].shape[0]
                
                main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
                wrist_key = 'robot0_eye_in_hand_image'
                
                # h5py 不支持重复索引，需去重
                unique_indices, inverse_indices = np.unique(global_indices, return_inverse=True)
                unique_main = demo_grp['obs'][main_key][unique_indices]
                unique_wrist = demo_grp['obs'][wrist_key][unique_indices]
                
                main_frames = unique_main[inverse_indices]
                wrist_frames = unique_wrist[inverse_indices]
                
                state_range_end = min(current_t + self.pred_horizon + 1, demo_len)
                state_raw = demo_grp['obs']['robot0_joint_pos'][current_t : state_range_end]
                
                if meta['has_teacher']:
                    teacher_siglip_tensor = torch.from_numpy(demo_grp['teacher_siglip'][current_t]).float()
                    feats = []
                    for offset in self.future_offsets:
                        target_idx = min(current_t + offset, demo_len - 1)
                        feats.append(torch.from_numpy(demo_grp['teacher_exo'][target_idx]).float())
                    future_exo_target = torch.stack(feats)

            print("从内存读取失败，改为硬盘读取。")

        # =========================================================
        # 🟡 公共处理逻辑 (Tensor转换与归一化)
        # =========================================================
        
        # 1. 视频归一化
        # [6, H, W, 3] -> [6, 3, H, W] -> Normalize
        main_t = torch.tensor(main_frames).float().permute(0, 3, 1, 2) / 255.0
        wrist_t = torch.tensor(wrist_frames).float().permute(0, 3, 1, 2) / 255.0
        
        main_t = self.normalize(main_t)
        wrist_t = self.normalize(wrist_t)
        
        # [2, 3, 6, H, W]
        video = torch.stack([main_t, wrist_t], dim=0).transpose(1, 2)

        # 2. 状态补齐与归一化
        target_len = self.pred_horizon + 1
        if state_raw.shape[0] < target_len:
            pad_len = target_len - state_raw.shape[0]
            # 使用最后一帧填充
            state_raw = np.concatenate([state_raw, np.tile(state_raw[-1:], (pad_len, 1))], axis=0)
            
        state_norm = (torch.tensor(state_raw).float() - self.action_mean) / self.action_std
        
        # 分离 Current State 和 Action Target
        # state_input_expanded: [6, 8] (复制当前状态以适配旧接口)
        state_input_expanded = state_norm[0].unsqueeze(0).repeat(self.window_size, 1)
        action_target = state_norm[1:] # [64, 8]

        # # 3. Anchor (First Frame)
        # curr_idx = int(demo_key.split('_')[1])
        # anchor_key = f"demo_{(curr_idx//5)*5}"
        
        # if anchor_key in self.anchor_bank:
        #     first_frame = self.anchor_bank[anchor_key]
        #     # 确保 Anchor 也被归一化 (如果 Bank 里存的是 Raw 0-1)
        #     # 这里的 anchor_bank 在 init 时已经 /255.0 了，但还没 Normalize
        #     # 简单起见，我们在使用时做 Normalize，或者确保 init 里不做
        #     # 根据 init 代码：self.anchor_bank 存的是 /255.0 后的。
        #     # 所以这里应用 Normalize
        #     first_frame = torch.stack([
        #         self.normalize(first_frame[0]), 
        #         self.normalize(first_frame[1])
        #     ], dim=0)
        # else:
        #     # Fallback (使用当前序列首帧)
        #     if self.in_memory:
        #         # 再次从 Cache 取首帧 (indices=0)
        #         m0 = torch.tensor(self.cache[demo_key]['main_img'][0]).float().permute(2, 0, 1) / 255.0
        #         w0 = torch.tensor(self.cache[demo_key]['wrist_img'][0]).float().permute(2, 0, 1) / 255.0
        #     else:
        #         # 极端情况的 fallback，暂不处理 h5py 打开，直接用当前 batch 的第一帧近似
        #         m0 = main_t[0] # 已经是 Norm 过的了
        #         w0 = wrist_t[0]
        #         # 注意：m0 w0 已经是 Normalized 的了，不需要再做
        #         first_frame = torch.stack([m0, w0], dim=0)
        #         # 跳过下面的 Normalize
            
        #     if 'first_frame' not in locals():
        #         first_frame = torch.stack([self.normalize(m0), self.normalize(w0)], dim=0)



        # ✅ 改为：无论是不是 Type A，都用自己的首帧
        # (保持你现有的 fallback 逻辑，并确保归一化)
        if self.in_memory:
            m0_raw = torch.tensor(self.cache[demo_key]['main_img'][0]).float().permute(2, 0, 1) / 255.0
            w0_raw = torch.tensor(self.cache[demo_key]['wrist_img'][0]).float().permute(2, 0, 1) / 255.0
        else:
            # ... 从 h5py 读取第 0 帧 ...
            pass

        first_frame = torch.stack([self.normalize(m0_raw), self.normalize(w0_raw)], dim=0)





        # 4. Teacher 默认值填充
        if teacher_siglip_tensor is None:
            teacher_siglip = torch.zeros(self.window_size, 1152)
            future_exo_target = torch.zeros(len(self.future_offsets), 1152)
        else:
            teacher_siglip = teacher_siglip_tensor.unsqueeze(0).repeat(self.window_size, 1)
            
        teacher_exo_legacy = torch.zeros(self.window_size, 1152)

        # 5. Tokenize Instruction
        text_tokens = self.tokenizer(meta['instruction'], return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids.squeeze(0)

        return {
            "video": video,
            "state": state_input_expanded,
            "action_target": action_target,
            "text_tokens": text_tokens,
            "first_frame": first_frame,
            "teacher_siglip": teacher_siglip,
            "teacher_exo": teacher_exo_legacy,
            "future_exo_target": future_exo_target
        }