# # ego单视角
# import torch
# import cv2
# import json
# import numpy as np
# from collections import deque
# from diffusers import DDIMScheduler
# import os
# from torch.amp import autocast
# from peft import LoraConfig, get_peft_model
# from transformers import T5Tokenizer
# import torch._dynamo
# from torchvision import transforms
# import time

# # === 导入你的模型 ===
# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper

# # === 基础路径配置 ===
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = "/yanghaochuan/data/124dataset_stats.json" 
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# STAGE_C_PATH = '/yanghaochuan/124checkpoints_finetune/StageC_ForeSight_step_7000.pt'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SafetyController:
#     def __init__(self):
#         self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.01
#         self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.01

#     def clip_actions(self, actions_batch):
#         actions_np = np.array(actions_batch)
#         joints = actions_np[:, :7]
#         gripper = actions_np[:, 7:]
#         joints_clipped = np.clip(joints, self.joint_limits_min, self.joint_limits_max)
#         return np.concatenate([joints_clipped, gripper], axis=1)

# class RealTimeAgent:
#     def __init__(self):
#         self.device = DEVICE
#         self.safety = SafetyController() 
#         self.pred_horizon = 64
#         self.trajectory_offset = None
        
#         # 🟢 [Alignment] 与 dataset_loader.py 保持一致
#         self.history_len = 500       # 模拟 dataset 中的 history_len
#         self.model_input_frames = 6  # 模拟 dataset 中的 window_size
        
#         self.debug_dir = f"debug_visuals_{int(time.time())}"
#         os.makedirs(self.debug_dir, exist_ok=True)
#         self.step_counter = 0

#         print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
#         except:
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
#         if not os.path.exists(STATS_PATH):
#             raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
#         with open(STATS_PATH, 'r') as f:
#             stats = json.load(f)
        
#         mean_raw = np.array(stats['action_mean'], dtype=np.float32)
#         std_raw = np.array(stats['action_std'], dtype=np.float32)
        
#         if mean_raw.shape[0] > 8:
#             self.action_mean = mean_raw[:8]
#             self.action_std = std_raw[:8]
#         elif mean_raw.shape[0] == 7:
#             self.action_mean = np.concatenate([mean_raw, [0.0]])
#             self.action_std = np.concatenate([std_raw, [1.0]])
#         else:
#             self.action_mean = mean_raw
#             self.action_std = std_raw
            
#         self.action_std = np.maximum(self.action_std, 1e-2)

#         # 🟢 [Alignment] 归一化参数与 VideoMAE/Dataset 一致
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                               std=[0.229, 0.224, 0.225])
#         self._init_models()
#         self._init_scheduler()
        
#         # 🟢 [Alignment] 历史 Buffer，对应 Dataset 中的 sliding window
#         self.video_buffer = deque(maxlen=self.history_len)
#         self.state_buffer = deque(maxlen=self.history_len)
        
#         self.first_frame_tensor = None
#         self.text_tokens = None 
#         self.default_prompt = "pick up the orange ball and put it on the plank"
        
#         self.warmup()

#     def _init_models(self):
#         print(f"[Agent] Initializing models on {self.device}...")
#         self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
#         self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
#         print(f"[Agent] Loading Checkpoint: {STAGE_C_PATH}")
#         ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        
#         peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
#         self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
#         if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
#         else: self.policy.load_state_dict(ckpt_c, strict=False)
        
#         # if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
#         if 'encoder_state_dict' in ckpt_c: 
#             print("正在加载 Encoder 权重...")
#             state_dict = ckpt_c['encoder_state_dict']
            
#             # 🛠️ 修复：移除编译或DDP产生的前缀
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 k_clean = k.replace("_orig_mod.", "").replace("module.", "")
#                 new_state_dict[k_clean] = v
            
#             # 🔍 诊断：不要用 strict=False，或者打印返回值
#             missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
            
#             if len(missing) > 0:
#                 print(f"⚠️ 警告：Encoder 加载有丢失键! (数量: {len(missing)})")
#                 print(f"   示例丢失: {missing[:5]}")
#             else:
#                 print("✅ Encoder 权重完美加载！")

#     def _init_scheduler(self):
#         self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
#         self.inference_steps = 25
#         self.scheduler.set_timesteps(self.inference_steps)

#     def warmup(self):
#         print("🔥 [System] Warming up model...")
#         dummy_video = torch.randn(1, 2, 3, 6, 224, 224, device=self.device, dtype=torch.bfloat16)
#         dummy_text = torch.randint(0, 1000, (1, 16), device=self.device)
#         dummy_state = torch.randn(1, 1, 8, device=self.device, dtype=torch.float32)
#         dummy_ff = torch.randn(1, 2, 3, 224, 224, device=self.device, dtype=torch.float32)
#         try:
#             with autocast('cuda', dtype=torch.bfloat16):
#                 feats = self.encoder(dummy_video, dummy_text, dummy_state, dummy_ff)
#                 feats["state"] = dummy_state[:, -1, :]
#                 latents = torch.randn(1, self.pred_horizon, 8, device=self.device)
#                 t = torch.tensor([0], device=self.device)
#                 _ = self.policy(latents, t, feats)
#             print("✅ Warmup done.")
#         except Exception as e:
#             print(f"❌ Warmup failed: {e}")

#     def preprocess_image(self, img_np):
#         resized = cv2.resize(img_np, (224, 224))
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         # 🟢 [Alignment] 必须归一化
#         tensor = self.normalize(tensor) 
#         return tensor

#     def save_model_input_visuals(self, vid_tensor, step_idx):
#         try:
#             wrist_t = vid_tensor[0, 1] 
#             mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).to(wrist_t.device)
#             std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).to(wrist_t.device)
#             wrist_t = wrist_t * std + mean
#             wrist_t = torch.clamp(wrist_t, 0, 1)
#             imgs = wrist_t.permute(1, 2, 3, 0).detach().cpu().numpy()
#             imgs = (imgs * 255).astype(np.uint8)
#             concat_img = np.hstack([imgs[i] for i in range(6)])
#             concat_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
#             save_path = os.path.join(self.debug_dir, f"step_{step_idx:04d}_buffer.jpg")
#             cv2.imwrite(save_path, concat_img)
#         except Exception as e:
#             print(f"⚠️ Visualization Failed: {e}")

#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session (Cold Start)...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         # 🟢 [Alignment] 首帧处理
#         wrist_tensor = self.preprocess_image(first_frame_img)
#         main_fake = torch.zeros_like(wrist_tensor)
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
#         self.text_tokens = tokens.to(self.device)
        
#         # Buffer 初始填入这一帧
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
#         self.video_buffer.append(video_frame_unit)
            
#         if current_qpos is None: current_qpos = np.zeros(8)
#         else: 
#             if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#             current_qpos = np.array(current_qpos, dtype=np.float32)
            
#         print(f"   🚩 [Reset QPos] {current_qpos[:7]} ... Grip: {current_qpos[7]}")
        
#         norm_qpos = (current_qpos - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos)
#         # === 🟢 添加这几行诊断代码 ===
#         print(f"\n🔍 [Stats Check] J0 Mean: {self.action_mean[0]:.4f}, Std: {self.action_std[0]:.4f}")
#         print(f"📉 [Input Norm Check] Current J0: {current_qpos[0]:.4f} -> Normalized: {norm_qpos[0]:.4f}")
#         if abs(norm_qpos[0]) > 3.0:
#             print("⚠️ 警告：初始状态严重偏离训练分布 (OOD)！模型可能会失效！")
#         # ============================
#         self.trajectory_offset = None  # 新增：确保每次新动作开始时重新计算对齐
#         print("[Agent] Trajectory offset reset.")

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         """
#         Stop-and-Think 模式:
#         1. 接收 frames_list (这些是机器人在执行上一个动作片段时捕获的‘历史’帧)
#         2. 将它们**全部**加入 Buffer (模拟时间流逝)
#         3. 进行均匀采样 (模拟 Training Loader)
#         4. 推理下一个动作
#         """
#         # ========================================================
#         # 🟢 Phase 1: Update History (Movement Phase Replay)
#         # ========================================================
#         # 将传入的所有帧按顺序加入 Buffer
#         # 这完全对应了训练集中，滑窗随着时间步 t 前进而前进
#         for frame in frames_list:
#             wrist_tensor = self.preprocess_image(frame)
#             main_fake = torch.zeros_like(wrist_tensor)
#             combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
#             self.video_buffer.append(combined_frame) 
        
#         # 🟢 State Update
#         # 我们假设这批图像对应的状态近似于当前状态 (或者你可以让Client传状态列表)
#         # 为了保证 Video/State Buffer 长度对齐，我们重复 append 当前状态
#         if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#         qpos_np = np.array(current_qpos, dtype=np.float32)
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
#         # 重复填充，使得状态历史长度与视觉历史长度匹配 (虽然模型只用最后一个)
#         for _ in range(len(frames_list)):
#             self.state_buffer.append(norm_qpos_np)
        
#         # ========================================================
#         # 🟢 Phase 2: Inference (Stop Phase)
#         # ========================================================
        
#         # 1. Uniform Sampling (完全复刻 Dataset __getitem__ 逻辑)
#         curr_len = len(self.video_buffer)
        
#         # np.linspace(0, valid_len-1, 6)
#         indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
#         selected_frames = [self.video_buffer[i] for i in indices]
        
#         # 构造 Batch
#         vid_t = torch.stack(selected_frames).to(self.device)
#         vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0) # [1, 2, 3, 6, H, W]

#         # 保存 Debug 图片 (确认模型到底看到了什么)
#         self.save_model_input_visuals(vid_t, self.step_counter)
#         self.step_counter += 1

#         # State: 取最新的 (FusionEncoder 只关注当前时刻)
#         state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
#         # 2. Diffusion Inference
#         self.scheduler.set_timesteps(self.inference_steps)
#         with autocast('cuda', dtype=torch.bfloat16):
#             features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
#             features["state"] = state_t[:, -1, :] 
#             latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#             for t in self.scheduler.timesteps:
#                 model_input = self.scheduler.scale_model_input(latents, t)
#                 t_tensor = torch.tensor([t], device=self.device)
#                 noise_pred = self.policy(model_input, t_tensor, features)
#                 latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
#         # 3. Denormalize & Output
#         normalized_actions = latents[0].float()
#         action_pred_np = normalized_actions.detach().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         # 夹爪二值化
#         # GRIPPER_OPEN_VAL = 0.0804  
#         # GRIPPER_CLOSE_VAL = 0.0428 
#         # GRIPPER_THRESHOLD = 0.0616 

#         # raw_gripper_pred = denormalized_actions[:, 7]
#         # binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
#         # denormalized_actions[:, 7] = binary_gripper

#         # if self.trajectory_offset is None:
#         #     # 计算模型预测的第 0 步与当前机器人真实位置的差值
#         #     # 只针对前 7 个关节 (J0-J6)
#         #     pred_start = denormalized_actions[0, :7]
#         #     real_start = qpos_np[:7]
#         #     self.trajectory_offset = pred_start - real_start
#         #     print(f"🚩 [Aligner] Offset calculated: {self.trajectory_offset}")
            
#         # # === 将打印逻辑移到这里 ===
#         # print(f"\n{'='*25} ALIGNED RDT Action (First 15 Steps) {'='*25}")
#         # header = f"{'Step':<4} | {'J0':^7} {'J1':^7} {'J2':^7} {'J3':^7} {'J4':^7} {'J5':^7} {'J6':^7} | {'Grip':^6}"
#         # print(header)
#         # for i in range(min(15, len(denormalized_actions))):
#         #     step_data = denormalized_actions[i]
#         #     joints_str = " ".join([f"{x: .4f}" for x in step_data[:7]])
#         #     print(f"{i:<4} | {joints_str} | {step_data[7]:.4f}")
#         # # ========================


#         # # 1. 获取实时位置 (qpos_np 是你在 step 开始时处理好的当前物理状态)
#         # real_start_pos = qpos_np[:8] 

#         # # 2. 强制覆盖 Step 0，确保物理层面绝对重合
#         # # 这样机器人执行第一个动作时就不会有任何“瞬跳”
#         # denormalized_actions[0, :8] = real_start_pos
        
#         # # 简单日志
#         # print(f"   >>> [Infer] BufferLen: {curr_len} | Pred J0: {denormalized_actions[0,0]:.3f}", end='\r')
        
#         # safe_actions = self.safety.clip_actions(denormalized_actions)
#         # return safe_actions.tolist()

#         # 2. 轨迹对齐逻辑 (Trajectory Aligner)
#         if self.trajectory_offset is None:
#             # 记录模型预测的起点与真实起点的偏差
#             # 注意：这里必须使用 .copy() 避免引用干扰
#             pred_start = denormalized_actions[0, :7].copy()
#             real_start = qpos_np[:7].copy()
#             self.trajectory_offset = pred_start - real_start
#             print(f"\n   🔧 [Aligner] Calibration Done. Offset J0: {self.trajectory_offset[0]:.4f}")

#         # 3. 应用对齐：减去全局偏差
#         denormalized_actions[:, :7] -= self.trajectory_offset

#         # 4. 【关键修复】物理强制覆盖 (Physical Overwrite)
#         # 无论模型预测和对齐计算结果如何，强制第一步绝对等于当前物理位置
#         # 这消除了所有计算残差，保证起步绝对平滑
#         denormalized_actions[0, :7] = qpos_np[:7]

#         # 5. 夹爪二值化处理
#         GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL, GRIPPER_THRESHOLD = 0.0804, 0.0428, 0.0616
#         raw_gripper_pred = denormalized_actions[:, 7]
#         denormalized_actions[:, 7] = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)

#         # 6. 【统一打印】在所有修正完成后再打印动作表
#         self._print_aligned_table(denormalized_actions)

#         # 7. 安全裁剪并返回
#         safe_actions = self.safety.clip_actions(denormalized_actions)
#         return safe_actions.tolist()

#     def _print_aligned_table(self, actions):
#         """辅助方法：打印最终发送给机械臂的动作序列"""
#         print(f"\n{'='*25} FINAL EXECUTABLE ACTION (Step 0-14) {'='*25}")
#         header = f"{'Step':<4} | {'J0':^7} {'J1':^7} {'J2':^7} {'J3':^7} {'J4':^7} {'J5':^7} {'J6':^7} | {'Grip':^6}"
#         print(header)
#         print("-" * 82)
#         for i in range(15):
#             joints = actions[i, :7]
#             print(f"{i:<4} | {' '.join([f'{x: .4f}' for x in joints])} | {actions[i, 7]:.4f}")
#         print("=" * 82 + "\n")



# # ego纯数值
# import torch
# import cv2
# import json
# import numpy as np
# from collections import deque
# from diffusers import DDIMScheduler
# import os
# from torch.amp import autocast
# from peft import LoraConfig, get_peft_model
# from transformers import T5Tokenizer
# from torchvision import transforms
# import time

# # 导入模型组件
# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper

# # 配置路径
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = "/yanghaochuan/data/124dataset_stats.json" 
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# STAGE_C_PATH = '/yanghaochuan/124checkpoints_finetune/StageC_ForeSight_step_7000.pt'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SafetyController:
#     def __init__(self):
#         # 关节限位保护
#         self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.01
#         self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.01

#     def clip_actions(self, actions_batch):
#         actions_np = np.array(actions_batch)
#         joints = actions_np[:, :7]
#         gripper = actions_np[:, 7:]
#         joints_clipped = np.clip(joints, self.joint_limits_min, self.joint_limits_max)
#         return np.concatenate([joints_clipped, gripper], axis=1)

# class RealTimeAgent:
#     def __init__(self):
#         self.device = DEVICE
#         self.safety = SafetyController() 
#         self.pred_horizon = 64
#         self.history_len = 500       
#         self.model_input_frames = 6  
        
#         # 1. 加载 Tokenizer 并预初始化 text_tokens
#         print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
#         self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
#         self.default_prompt = "pick up the orange ball and put it on the plank"
        
#         # 🟢 修复 ValueError: 确保在初始化时就生成 text_tokens
#         self.text_tokens = self.tokenizer(
#             self.default_prompt, 
#             return_tensors="pt", 
#             padding="max_length", 
#             max_length=16, 
#             truncation=True
#         ).input_ids.to(self.device)

#         # 2. 归一化参数加载
#         if not os.path.exists(STATS_PATH):
#             raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
#         with open(STATS_PATH, 'r') as f:
#             stats = json.load(f)
#         self.action_mean = np.array(stats['action_mean'][:8], dtype=np.float32)
#         self.action_std = np.maximum(np.array(stats['action_std'][:8], dtype=np.float32), 1e-2)

#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
#         # 3. 初始化模型结构
#         self._init_models()
#         self._init_scheduler()
        
#         self.video_buffer = deque(maxlen=self.history_len)
#         self.state_buffer = deque(maxlen=self.history_len)
#         self.first_frame_tensor = None
        
#         # 4. 执行预热
#         self.warmup()

#     def _init_models(self):
#         print(f"[Agent] Initializing models on {self.device}...")
#         self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
#         self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
#         print(f"[Agent] Loading Checkpoint: {STAGE_C_PATH}")
#         ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        
#         # LoRA 配置
#         peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
#         self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
#         if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
#         else: self.policy.load_state_dict(ckpt_c, strict=False)
        
#         if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)

#     def _init_scheduler(self):
#         self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
#         self.scheduler.set_timesteps(25)

#     def warmup(self):
#         """🟢 修复 AttributeError: 确保类定义内包含此方法"""
#         print("🔥 [System] Warming up model...")
#         dummy_video = torch.randn(1, 2, 3, 6, 224, 224, device=self.device, dtype=torch.bfloat16)
#         dummy_state = torch.randn(1, 1, 8, device=self.device, dtype=torch.float32)
#         dummy_ff = torch.randn(1, 2, 3, 224, 224, device=self.device, dtype=torch.float32)
#         try:
#             with autocast('cuda', dtype=torch.bfloat16):
#                 # 使用已经初始化好的 self.text_tokens
#                 feats = self.encoder(dummy_video, self.text_tokens, dummy_state, dummy_ff)
#                 feats["state"] = dummy_state[:, -1, :]
#                 latents = torch.randn(1, self.pred_horizon, 8, device=self.device)
#                 t_tensor = torch.tensor([0], device=self.device)
#                 _ = self.policy(latents, t_tensor, feats)
#             print("✅ Warmup done.")
#         except Exception as e:
#             print(f"❌ Warmup failed: {e}")

#     def preprocess_image(self, img_np):
#         resized = cv2.resize(img_np, (224, 224))
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         return self.normalize(tensor)

#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         wrist_tensor = self.preprocess_image(first_frame_img)
#         main_fake = torch.zeros_like(wrist_tensor)
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         # 更新 text_tokens（如果 prompt 改变）
#         self.text_tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids.to(self.device)
        
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
#         self.video_buffer.append(video_frame_unit)
        
#         if current_qpos is None: current_qpos = np.zeros(8)
#         norm_qpos = (np.array(current_qpos[:8]) - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos)

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         # 1. 更新 Buffer
#         for frame in frames_list:
#             wrist_t = self.preprocess_image(frame)
#             self.video_buffer.append(torch.stack([torch.zeros_like(wrist_t), wrist_t], dim=0)) 
        
#         qpos_np = np.array(current_qpos[:8], dtype=np.float32)
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos_np)
        
#         # 2. 采样输入
#         curr_len = len(self.video_buffer)
#         indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
#         vid_t = torch.stack([self.video_buffer[i] for i in indices]).to(self.device).permute(1, 2, 0, 3, 4).unsqueeze(0)
#         state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).view(1, 1, 8).to(self.device)
        
#         # 3. 推理
#         with autocast('cuda', dtype=torch.bfloat16):
#             # 🟢 此时 self.text_tokens 已在 __init__ 确保非空
#             features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
#             features["state"] = state_t[:, -1, :] 
#             latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#             for t in self.scheduler.timesteps:
#                 model_input = self.scheduler.scale_model_input(latents, t)
#                 t_tensor = torch.tensor([t], device=self.device)
#                 noise_pred = self.policy(model_input, t_tensor, features)
#                 latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
#         # 4. 反归一化
#         action_pred_np = latents[0].float().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         # 夹爪二值化
#         GRIPPER_THRESHOLD = 0.0616
#         denormalized_actions[:, 7] = np.where(denormalized_actions[:, 7] > GRIPPER_THRESHOLD, 0.0804, 0.0428)
        
#         # 安全裁剪并返回
#         return self.safety.clip_actions(denormalized_actions).tolist()




# #ego 二值化
# import torch
# import cv2
# import json
# import numpy as np
# from collections import deque
# from diffusers import DDIMScheduler
# import os
# from torch.amp import autocast
# from peft import LoraConfig, get_peft_model
# from transformers import T5Tokenizer
# from torchvision import transforms
# import time

# # 导入模型组件
# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper

# # 配置路径
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = "/yanghaochuan/data/40dataset_stats.json" 
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# STAGE_C_PATH = '/yanghaochuan/22checkpoints_finetune/StageC_ForeSight_step_3000.pt'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SafetyController:
#     def __init__(self):
#         # 关节限位保护
#         self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.01
#         self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.01

#     def clip_actions(self, actions_batch):
#         actions_np = np.array(actions_batch)
#         joints = actions_np[:, :7]
#         gripper = actions_np[:, 7:]
#         joints_clipped = np.clip(joints, self.joint_limits_min, self.joint_limits_max)
#         return np.concatenate([joints_clipped, gripper], axis=1)

# class RealTimeAgent:
#     def __init__(self):
#         self.device = DEVICE
#         self.safety = SafetyController() 
#         self.pred_horizon = 64
#         self.history_len = 500       
#         self.model_input_frames = 6  
        
#         # 🟢 [新增] 定义用于输入转换的物理阈值
#         # 这是物理世界中判断开闭的界限 (根据你的 stats 文件)
#         self.PHYSICAL_GRIPPER_THRESHOLD = 0.0616 

#         # 1. 加载 Tokenizer 并预初始化 text_tokens
#         print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
#         except:
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
            
#         self.default_prompt = "pick up the orange ball and put it on the plank"
        
#         self.text_tokens = self.tokenizer(
#             self.default_prompt, 
#             return_tensors="pt", 
#             padding="max_length", 
#             max_length=16, 
#             truncation=True
#         ).input_ids.to(self.device)

#         # 2. 归一化参数加载
#         if not os.path.exists(STATS_PATH):
#             raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
#         with open(STATS_PATH, 'r') as f:
#             stats = json.load(f)
#         self.action_mean = np.array(stats['action_mean'][:8], dtype=np.float32)
#         self.action_std = np.maximum(np.array(stats['action_std'][:8], dtype=np.float32), 1e-2)

#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
#         # 3. 初始化模型结构
#         self._init_models()
#         self._init_scheduler()
        
#         self.video_buffer = deque(maxlen=self.history_len)
#         self.state_buffer = deque(maxlen=self.history_len)
#         self.first_frame_tensor = None
        
#         # 4. 执行预热
#         self.warmup()

#     # def _init_models(self):
#     #     print(f"[Agent] Initializing models on {self.device}...")
#     #     self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
#     #     self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
#     #     print(f"[Agent] Loading Checkpoint: {STAGE_C_PATH}")
#     #     ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        
#     #     # LoRA 配置
#     #     peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
#     #     self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
#     #     if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
#     #     else: self.policy.load_state_dict(ckpt_c, strict=False)
        
#     #     if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)

#     def _init_models(self):
#         print(f"[Agent] Initializing models on {self.device}...")
#         self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
        
#         # ... (RDT 初始化代码不变) ...
#         self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
#         # LoRA 配置 (不变)
#         peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
#         self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)

#         print(f"[Agent] Loading Checkpoint: {STAGE_C_PATH}")
#         ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        
#         # === 🔍 严谨的加载检查代码 ===
        
#         # 1. 准备 State Dict
#         if 'rdt_state_dict' in ckpt_c:
#             rdt_state_dict = ckpt_c['rdt_state_dict']
#         else:
#             rdt_state_dict = ckpt_c
            
#         # 2. 加载并捕获返回结果 (strict=False)
#         # load_result 是一个 namedtuple: (missing_keys, unexpected_keys)
#         load_result = self.policy.load_state_dict(rdt_state_dict, strict=False)
        
#         # 3. 打印分析
#         missing = load_result.missing_keys
#         unexpected = load_result.unexpected_keys
        
#         print("\n" + "="*50)
#         print("🧐 Checkpoint Loading Inspection")
#         print("="*50)
        
#         # 检查 LoRA 是否加载
#         lora_keys = [k for k in missing if 'lora' in k]
#         if len(lora_keys) > 0:
#             print(f"❌ 警告! LoRA 参数未加载 (Missing {len(lora_keys)} keys):")
#             print(f"   Example: {lora_keys[0]}")
#         else:
#             print("✅ LoRA 参数已成功加载。")

#         # 检查 Head (输出层) 是否加载
#         # 通常 RDT 的输出头包含 'head' 或 'final_layer'，根据具体模型结构调整
#         # 这里假设最后一层包含 'linear' 或特定名称
#         head_missing = [k for k in missing if 'head' in k or 'out_proj' in k] # 举例
#         if len(head_missing) > 0 and len(head_missing) < 10: # 允许少量缺失，但不能全缺
#              print(f"⚠️ 注意: 部分输出层参数缺失: {head_missing}")
        
#         # 检查 Encoder
#         if 'encoder_state_dict' in ckpt_c: 
#             self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
#             print("✅ Encoder (Adapter) weights loaded.")
#         else:
#             print("⚠️ Warning: No encoder_state_dict found in checkpoint!")

#         print("="*50 + "\n")

#     def _init_scheduler(self):
#         self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
#         self.scheduler.set_timesteps(25)

#     def warmup(self):
#         print("🔥 [System] Warming up model...")
#         dummy_video = torch.randn(1, 2, 3, 6, 224, 224, device=self.device, dtype=torch.bfloat16)
#         dummy_state = torch.randn(1, 1, 8, device=self.device, dtype=torch.float32)
#         dummy_ff = torch.randn(1, 2, 3, 224, 224, device=self.device, dtype=torch.float32)
#         try:
#             with autocast('cuda', dtype=torch.bfloat16):
#                 feats = self.encoder(dummy_video, self.text_tokens, dummy_state, dummy_ff)
#                 feats["state"] = dummy_state[:, -1, :]
#                 latents = torch.randn(1, self.pred_horizon, 8, device=self.device)
#                 t_tensor = torch.tensor([0], device=self.device)
#                 _ = self.policy(latents, t_tensor, feats)
#             print("✅ Warmup done.")
#         except Exception as e:
#             print(f"❌ Warmup failed: {e}")

#     def preprocess_image(self, img_np):
#         resized = cv2.resize(img_np, (224, 224))
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         return self.normalize(tensor)

#     # 🟢 [新增] 辅助函数：处理输入的物理状态
#     def _preprocess_qpos_for_model(self, current_qpos):
#         """
#         将机器人的物理状态转换为模型理解的状态。
#         特别是将夹爪的物理数值 (0.04~0.08) 转换为训练时的二值 (1.0/-1.0)
#         """
#         if current_qpos is None: 
#             qpos_new = np.zeros(8, dtype=np.float32)
#         else:
#             qpos_new = np.array(current_qpos, dtype=np.float32).copy()
#             # 补齐维度
#             if len(qpos_new) == 7: 
#                 qpos_new = np.concatenate([qpos_new, [0.0]])
#             elif len(qpos_new) > 8:
#                  qpos_new = qpos_new[:8]
        
#         # === 关键转换 ===
#         # 如果物理值 > 0.0616，模型应该看到 1.0 (Open)
#         # 如果物理值 < 0.0616，模型应该看到 -1.0 (Close)
#         raw_gripper = qpos_new[7]
#         if raw_gripper > self.PHYSICAL_GRIPPER_THRESHOLD:
#             qpos_new[7] = 1.0
#         else:
#             qpos_new[7] = -1.0
            
#         return qpos_new

#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         # 1. 图像处理
#         wrist_tensor = self.preprocess_image(first_frame_img)
#         main_fake = torch.zeros_like(wrist_tensor)
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         # 更新 text_tokens
#         self.text_tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids.to(self.device)
        
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
#         self.video_buffer.append(video_frame_unit)
        
#         # 2. 状态处理 [修改点]
#         # 先将物理状态转为二值化状态，再进行归一化
#         model_input_qpos = self._preprocess_qpos_for_model(current_qpos)
        
#         print(f"   🚩 [Reset QPos] {current_qpos[:8]}")
#         print(f"   🚩 [Input Check] Raw Grip: {current_qpos[7] if current_qpos is not None else 0:.4f} -> Model Input: {model_input_qpos[7]:.1f}")
        
#         norm_qpos = (model_input_qpos - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos)

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         # 1. 更新图像 Buffer
#         for frame in frames_list:
#             wrist_t = self.preprocess_image(frame)
#             self.video_buffer.append(torch.stack([torch.zeros_like(wrist_t), wrist_t], dim=0)) 
        
#         # 2. 更新状态 Buffer [修改点]
#         # 同样，必须先二值化，再归一化
#         model_input_qpos = self._preprocess_qpos_for_model(current_qpos)
#         norm_qpos_np = (model_input_qpos - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos_np)
        
#         # 3. 采样输入
#         curr_len = len(self.video_buffer)
#         indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
#         vid_t = torch.stack([self.video_buffer[i] for i in indices]).to(self.device).permute(1, 2, 0, 3, 4).unsqueeze(0)
#         state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).view(1, 1, 8).to(self.device)
        
#         # 4. 推理
#         with autocast('cuda', dtype=torch.bfloat16):
#             features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
#             features["state"] = state_t[:, -1, :] 
#             latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#             for t in self.scheduler.timesteps:
#                 model_input = self.scheduler.scale_model_input(latents, t)
#                 t_tensor = torch.tensor([t], device=self.device)
#                 noise_pred = self.policy(model_input, t_tensor, features)
#                 latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
#         # 5. 反归一化
#         action_pred_np = latents[0].float().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         # === 🟢 [核心修改] 输出二值化逻辑 ===
#         # 模型预测值 > 0.0 -> 输出 1.0 (Open)
#         # 模型预测值 <= 0.0 -> 输出 -1.0 (Close)
#         # 这样客户端接收到明确的信号，不会收到 0.3 这种中间值
#         raw_gripper_pred = denormalized_actions[:, 7]
#         denormalized_actions[:, 7] = np.where(raw_gripper_pred > 0.0, 1.0, -1.0)
        
#         # 安全裁剪并返回
#         return self.safety.clip_actions(denormalized_actions).tolist()




#ego 双摄二值化爪夹

import torch
import cv2
import json
import numpy as np
from collections import deque
from diffusers import DDIMScheduler
import os
from torch.amp import autocast
from peft import LoraConfig, get_peft_model
from transformers import T5Tokenizer
from torchvision import transforms
import time

# 导入模型组件
from model.fusion_encoder import FusionEncoder
from model.rdt_model import RDTWrapper

# 配置路径
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = "/yanghaochuan/data/23dataset_stats.json" 
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# 确保这里加载的是您最新的 Stage C 权重
STAGE_C_PATH = '/yanghaochuan/23checkpoints_finetune/StageC_ForeSight_step_4000.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafetyController:
    def __init__(self):
        # 关节限位保护
        self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.01
        self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.01

    def clip_actions(self, actions_batch):
        actions_np = np.array(actions_batch)
        joints = actions_np[:, :7]
        gripper = actions_np[:, 7:]
        joints_clipped = np.clip(joints, self.joint_limits_min, self.joint_limits_max)
        return np.concatenate([joints_clipped, gripper], axis=1)

class RealTimeAgent:
    def __init__(self):
        self.device = DEVICE
        self.safety = SafetyController() 
        self.pred_horizon = 64
        self.history_len = 500       
        self.model_input_frames = 6  
        
        # 🟢 定义用于输入转换的物理阈值
        self.PHYSICAL_GRIPPER_THRESHOLD = 0.0616 

        # 1. 加载 Tokenizer 并预初始化 text_tokens
        print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        except:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
            
        self.default_prompt = "pick up the orange ball and put it on the plank"
        
        self.text_tokens = self.tokenizer(
            self.default_prompt, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=16, 
            truncation=True
        ).input_ids.to(self.device)

        # 2. 归一化参数加载
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        self.action_mean = np.array(stats['action_mean'][:8], dtype=np.float32)
        self.action_std = np.maximum(np.array(stats['action_std'][:8], dtype=np.float32), 1e-2)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 3. 初始化模型结构
        self._init_models()
        self._init_scheduler()
        
        self.video_buffer = deque(maxlen=self.history_len)
        self.state_buffer = deque(maxlen=self.history_len)
        self.first_frame_tensor = None
        
        # 4. 执行预热
        self.warmup()

    def _init_models(self):
        print(f"[Agent] Initializing models on {self.device}...")
        # 🟢 确保 teacher_dim 与训练一致
        self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
        
        self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
        # LoRA 配置
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
        self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)

        print(f"[Agent] Loading Checkpoint: {STAGE_C_PATH}")
        ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        
        # 1. 加载 RDT
        if 'rdt_state_dict' in ckpt_c:
            rdt_state_dict = ckpt_c['rdt_state_dict']
        else:
            rdt_state_dict = ckpt_c
        
        load_result = self.policy.load_state_dict(rdt_state_dict, strict=False)
        print(f"RDT Loaded. Missing keys: {len(load_result.missing_keys)}")

        # 2. 加载 Encoder
        if 'encoder_state_dict' in ckpt_c: 
            self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
            print("✅ Encoder (Adapter) weights loaded.")
        else:
            print("⚠️ Warning: No encoder_state_dict found in checkpoint!")

    def _init_scheduler(self):
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
        self.scheduler.set_timesteps(25)

    def warmup(self):
        print("🔥 [System] Warming up model...")
        # 🟢 Warmup 输入维度必须是双摄: [1, 2, 3, 6, 224, 224]
        dummy_video = torch.randn(1, 2, 3, 6, 224, 224, device=self.device, dtype=torch.bfloat16)
        dummy_state = torch.randn(1, 1, 8, device=self.device, dtype=torch.float32)
        dummy_ff = torch.randn(1, 2, 3, 224, 224, device=self.device, dtype=torch.float32)
        try:
            with autocast('cuda', dtype=torch.bfloat16):
                feats = self.encoder(dummy_video, self.text_tokens, dummy_state, dummy_ff)
                feats["state"] = dummy_state[:, -1, :]
                latents = torch.randn(1, self.pred_horizon, 8, device=self.device)
                t_tensor = torch.tensor([0], device=self.device)
                _ = self.policy(latents, t_tensor, feats)
            print("✅ Warmup done.")
        except Exception as e:
            print(f"❌ Warmup failed: {e}")

    def preprocess_image(self, img_np):
        """Standard ImageNet Normalization"""
        if img_np is None: return torch.zeros(3, 224, 224, dtype=torch.float32)
        resized = cv2.resize(img_np, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return self.normalize(tensor)

    def _preprocess_qpos_for_model(self, current_qpos):
        """将物理状态转换为模型训练时的状态格式 (夹爪二值化)"""
        if current_qpos is None: 
            qpos_new = np.zeros(8, dtype=np.float32)
        else:
            qpos_new = np.array(current_qpos, dtype=np.float32).copy()
            if len(qpos_new) == 7: 
                qpos_new = np.concatenate([qpos_new, [0.0]])
            elif len(qpos_new) > 8:
                 qpos_new = qpos_new[:8]
        
        # 夹爪数值 -> 模型逻辑值 (1.0/-1.0)
        raw_gripper = qpos_new[7]
        if raw_gripper > self.PHYSICAL_GRIPPER_THRESHOLD:
            qpos_new[7] = 1.0 # Open
        else:
            qpos_new[7] = -1.0 # Close
            
        return qpos_new

    def reset_session(self, first_main_img, first_wrist_img, current_qpos=None):
        """
        🟢 [双摄修改] 重置 Session，同时接收 Main 和 Wrist 的首帧
        """
        print("[Agent] Resetting session (Dual View)...")
        self.video_buffer.clear()
        self.state_buffer.clear()
        
        # 1. 双摄图像处理
        main_tensor = self.preprocess_image(first_main_img)
        wrist_tensor = self.preprocess_image(first_wrist_img)
        
        # Stack: [2, 3, 224, 224] (Main, Wrist)
        self.first_frame_tensor = torch.stack([main_tensor, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
        # 更新 prompt token
        self.text_tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids.to(self.device)
        
        # 填入 Buffer (Shape: [2, 3, 224, 224])
        video_frame_unit = torch.stack([main_tensor, wrist_tensor], dim=0)
        self.video_buffer.append(video_frame_unit)
        
        # 2. 状态处理
        model_input_qpos = self._preprocess_qpos_for_model(current_qpos)
        print(f"   🚩 [Reset QPos] {current_qpos[:7]}")
        print(f"   🚩 [Grip Check] Raw: {current_qpos[7] if current_qpos is not None else 0:.4f} -> Model: {model_input_qpos[7]:.1f}")
        
        norm_qpos = (model_input_qpos - self.action_mean) / self.action_std
        self.state_buffer.append(norm_qpos)

    @torch.no_grad()
    def step(self, main_frames_list, wrist_frames_list, current_qpos):
        """
        🟢 [双摄修改] Step 接受两个图像列表 (Main & Wrist)
        """
        # 1. 更新图像 Buffer (假设两个列表长度一致)
        # 如果长度不一致，以较短的为准，或根据实际业务逻辑调整
        n_frames = min(len(main_frames_list), len(wrist_frames_list))
        
        for i in range(n_frames):
            main_t = self.preprocess_image(main_frames_list[i])
            wrist_t = self.preprocess_image(wrist_frames_list[i])
            # Stack into [2, 3, 224, 224] and append
            self.video_buffer.append(torch.stack([main_t, wrist_t], dim=0))
        
        # 2. 更新状态 Buffer
        model_input_qpos = self._preprocess_qpos_for_model(current_qpos)
        norm_qpos_np = (model_input_qpos - self.action_mean) / self.action_std
        self.state_buffer.append(norm_qpos_np)
        
        # 3. 采样 (Sliding Window)
        curr_len = len(self.video_buffer)
        indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
        
        # Stack frames: [T, 2, 3, H, W] -> Permute to [1, 2, 3, T, H, W]
        # 这里的 T=6 (model_input_frames)
        # 最终输入给 Encoder 的形状: [1, 2, 3, 6, 224, 224]
        vid_t = torch.stack([self.video_buffer[i] for i in indices]).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)
        
        state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).view(1, 1, 8).to(self.device)
        
        # 4. 推理
        with autocast('cuda', dtype=torch.bfloat16):
            features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            features["state"] = state_t[:, -1, :] 
            latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
            for t in self.scheduler.timesteps:
                model_input = self.scheduler.scale_model_input(latents, t)
                t_tensor = torch.tensor([t], device=self.device)
                noise_pred = self.policy(model_input, t_tensor, features)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. 反归一化 & 输出处理
        action_pred_np = latents[0].float().cpu().numpy()
        denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
        # 夹爪输出二值化 ( >0 -> Open, <=0 -> Close)
        raw_gripper_pred = denormalized_actions[:, 7]
        denormalized_actions[:, 7] = np.where(raw_gripper_pred > 0.0, 1.0, -1.0)
        
        # # 安全裁剪
        # return self.safety.clip_actions(denormalized_actions).tolist()


        # 6. 【统一打印】在所有修正完成后再打印动作表
        self._print_aligned_table(denormalized_actions)

        # 7. 安全裁剪并返回
        safe_actions = self.safety.clip_actions(denormalized_actions)
        return safe_actions.tolist()

    def _print_aligned_table(self, actions):
        """辅助方法：打印最终发送给机械臂的动作序列"""
        print(f"\n{'='*25} FINAL EXECUTABLE ACTION (Step 0-14) {'='*25}")
        header = f"{'Step':<4} | {'J0':^7} {'J1':^7} {'J2':^7} {'J3':^7} {'J4':^7} {'J5':^7} {'J6':^7} | {'Grip':^6}"
        print(header)
        print("-" * 82)
        for i in range(15):
            joints = actions[i, :7]
            print(f"{i:<4} | {' '.join([f'{x: .4f}' for x in joints])} | {actions[i, 7]:.4f}")
        print("=" * 82 + "\n")