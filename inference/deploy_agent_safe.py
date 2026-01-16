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

# # === 导入你的模型 ===
# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper

# # === 基础路径配置 ===
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = "/yanghaochuan/data/13dataset_stats.json"
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# STAGE_C_PATH = '/yanghaochuan/checkpoints/12stageC_step_4000.pt'

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
        
#         self._init_models()
#         self._init_scheduler()
        
#         self.window_size = 16
#         self.video_buffer = deque(maxlen=self.window_size)
#         self.state_buffer = deque(maxlen=self.window_size)
#         self.first_frame_tensor = None
#         self.text_tokens = None 
#         self.default_prompt = "pick up the orange ball"

#     def _init_models(self):
#         # ... (模型初始化代码保持不变) ...
#         print(f"[Agent] Initializing models on {self.device}...")
#         self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
#         self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
#         ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
#         peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
#         self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
#         if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
#         else: self.policy.load_state_dict(ckpt_c, strict=False)
        
#         if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
#         else: raise ValueError(f"❌ No encoder_state_dict in {STAGE_C_PATH}")
        
#         torch._dynamo.config.suppress_errors = True
#         try: self.encoder = torch.compile(self.encoder, mode="default")
#         except: pass

#     def _init_scheduler(self):
#         self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
#         self.inference_steps = 25
#         self.scheduler.set_timesteps(self.inference_steps)

#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session (Cold Start)...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         ff_resized = cv2.resize(first_frame_img, (224, 224))
#         ff_rgb = cv2.cvtColor(ff_resized, cv2.COLOR_BGR2RGB)
#         wrist_tensor = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         main_fake = torch.zeros_like(wrist_tensor)
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
#         self.text_tokens = tokens.to(self.device)
        
#         # Buffer 初始化 (虽然马上会被 step 覆盖，但为了安全先填满)
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0) 
#         for _ in range(self.window_size):
#             self.video_buffer.append(video_frame_unit) 
            
#         if current_qpos is None: current_qpos = np.zeros(8)
#         else: 
#             if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#             current_qpos = np.array(current_qpos, dtype=np.float32)
#         norm_qpos = (current_qpos - self.action_mean) / self.action_std
#         for _ in range(self.window_size):
#             self.state_buffer.append(norm_qpos)

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         """
#         :param frames_list: 包含 16 帧真实历史图像的列表 (List[np.array])
#         :param current_qpos: 当前机器人关节状态
#         """
#         # =========================================================================
#         # [逻辑修正] 完全重置 Buffer，填入真实的 16 帧历史
#         # =========================================================================
#         self.video_buffer.clear()
        
#         for frame in frames_list:
#             # 预处理每一帧
#             frame_resized = cv2.resize(frame, (224, 224))
#             frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#             wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
#             main_fake = torch.zeros_like(wrist_tensor)
#             combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
            
#             self.video_buffer.append(combined_frame)
        
#         # 确保填满了 (如果客户端传来的不足16帧，应该在客户端补齐，但这里双重保险)
#         while len(self.video_buffer) < self.window_size:
#             self.video_buffer.append(self.video_buffer[-1])

#         # 2. State Preprocess
#         if len(current_qpos) == 7:
#             current_qpos = list(current_qpos) + [0.0]
        
#         qpos_np = np.array(current_qpos, dtype=np.float32)
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
#         # 状态 Buffer 也应该刷新，但通常我们只有当前状态
#         # 策略：假设过去16帧的状态都近似于当前状态 (或者你可以让客户端也传状态历史)
#         # 这里简化处理：填满当前状态
#         self.state_buffer.clear()
#         for _ in range(self.window_size):
#             self.state_buffer.append(norm_qpos_np)
        
#         # 3. Batch Construction
#         vid_t = torch.stack(list(self.video_buffer)).to(self.device)
#         vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0) # [1, 2, 3, 16, 224, 224]
        
#         state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         # 4. Inference
#         self.scheduler.set_timesteps(self.inference_steps)
#         # with autocast('cuda', dtype=torch.bfloat16):
#         #     features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
#         #     latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#         #     for t in self.scheduler.timesteps:
#         #         model_input = self.scheduler.scale_model_input(latents, t)
#         #         t_tensor = torch.tensor([t], device=self.device)
#         #         noise_pred = self.policy(model_input, t_tensor, features)
#         #         latents = self.scheduler.step(noise_pred, t, latents).prev_sample
#         with autocast('cuda', dtype=torch.bfloat16):
#             # (1) 获取视觉特征
#             features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            
#             # =================================================================
#             # 🚨 [关键修复] 手动注入 State！
#             # 必须与训练时的 behavior 一致：取时间窗口的最后一帧 state[:, -1, :]
#             # =================================================================
#             features["state"] = state_t[:, -1, :] 
            
#             latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#             for t in self.scheduler.timesteps:
#                 model_input = self.scheduler.scale_model_input(latents, t)
#                 t_tensor = torch.tensor([t], device=self.device)
                
#                 # (2) 传入包含 state 的完整字典
#                 noise_pred = self.policy(model_input, t_tensor, features)
                
#                 latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
#         # ... (后续代码不变) ...
            
#         normalized_actions = latents[0].float()
#         action_pred_np = normalized_actions.detach().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         gripper_val = denormalized_actions[0, 7]
#         print(f"   >>> [Model Output] Gripper: {gripper_val:.4f} (Threshold: <0.06 Close)", end='\r')
#         safe_actions = self.safety.clip_actions(denormalized_actions)
#         return safe_actions.tolist()

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

# # === 导入你的模型 ===
# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper

# # === 基础路径配置 ===
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# # 使用新的 16dataset_stats (对应新的采样策略)
# STATS_PATH = "/yanghaochuan/data/111dataset_stats.json"
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# # 使用 ForeSight 训练出的 Checkpoint
# STAGE_C_PATH = '/yanghaochuan/114checkpoints_finetune/StageC_ForeSight_step_4000.pt'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class SafetyController:
#     def __init__(self):
#         # Franka 关节极限 (安全余量)
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

#         # === 🟢 ForeSight 核心参数 ===
#         self.history_len = 500       # Buffer 长度：覆盖过去 2-3 秒
#         self.model_input_frames = 6 # 模型实际输入：均匀采样 6 帧
#         # ===========================

#         print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
#         except:
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
#         # 加载统计数据
#         if not os.path.exists(STATS_PATH):
#             raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
#         with open(STATS_PATH, 'r') as f:
#             stats = json.load(f)
        
#         mean_raw = np.array(stats['action_mean'], dtype=np.float32)
#         std_raw = np.array(stats['action_std'], dtype=np.float32)
        
#         # 维度修正
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

#         # 🟢 [新增] 归一化 (与训练完全一致)
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                               std=[0.229, 0.224, 0.225])
#         self._init_models()
#         self._init_scheduler()
        
#         # 初始化 Buffer (长度为 history_len)
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
        
#         if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
        
#         torch._dynamo.config.suppress_errors = True
#         try: self.encoder = torch.compile(self.encoder, mode="default")
#         except: pass

#     def _init_scheduler(self):
#         self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
#         self.inference_steps = 25
#         self.scheduler.set_timesteps(self.inference_steps)


#     # 🟢 [新增] 预热函数
#     def warmup(self):
#         print("🔥 [System] Warming up model (compilation)... This may take 1 min.")
#         # 构造假的输入 (Batch=1, View=2, Channel=3, Time=6, H=224, W=224)
#         dummy_video = torch.randn(1, 2, 3, 6, 224, 224, device=self.device, dtype=torch.bfloat16)
#         dummy_text = torch.randint(0, 1000, (1, 16), device=self.device)
#         dummy_state = torch.randn(1, 1, 8, device=self.device, dtype=torch.float32)
#         dummy_ff = torch.randn(1, 2, 3, 224, 224, device=self.device, dtype=torch.float32)
        
#         try:
#             with autocast('cuda', dtype=torch.bfloat16):
#                 # 跑一次 Encoder
#                 feats = self.encoder(dummy_video, dummy_text, dummy_state, dummy_ff)
#                 feats["state"] = dummy_state[:, -1, :]
#                 # 跑一次 Policy
#                 latents = torch.randn(1, self.pred_horizon, 8, device=self.device)
#                 t = torch.tensor([0], device=self.device)
#                 _ = self.policy(latents, t, feats)
#             print("✅ Warmup done. Ready to serve.")
#         except Exception as e:
#             print(f"❌ Warmup failed: {e}")

#     # 🟢 [新增] 图像预处理 (暴力 Resize + 归一化)
#     def preprocess_image(self, img_np):
#         # 1. 暴力 Resize: 1280x720 -> 224x224
#         # 这会产生畸变，但保留了所有边缘信息，且与你的训练数据预处理(preprocess_with_teachers.py)一致
#         resized = cv2.resize(img_np, (224, 224))
        
#         # 2. BGR -> RGB
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
#         # 3. To Tensor & Normalize
#         tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         tensor = self.normalize(tensor) # <--- 关键！
        
#         return tensor


#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session (Cold Start)...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         # 处理首帧 (Anchor)
#         ff_resized = cv2.resize(first_frame_img, (224, 224))
#         ff_rgb = cv2.cvtColor(ff_resized, cv2.COLOR_BGR2RGB)
#         # wrist_tensor = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         # 1. 基础转换
#         wrist_tensor_raw = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
#         # 2. 🟢 [修复] 必须加上归一化！
#         wrist_tensor = self.normalize(wrist_tensor_raw)
#         main_fake = torch.zeros_like(wrist_tensor)
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
#         self.text_tokens = tokens.to(self.device)
        
#         # # 填满 buffer (冷启动填充)
#         # # 注意：这里我们填满 history_len，这样初始采样就是全是首帧
#         # video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0) 
#         # for _ in range(self.history_len):
#         #     self.video_buffer.append(video_frame_unit) 

#         # 🟢 [修改] 动态 Buffer 策略
#         # 只存入当前这 1 帧。绝不填充 500 次！
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
#         self.video_buffer.append(video_frame_unit)
            
#         if current_qpos is None: current_qpos = np.zeros(8)
#         else: 
#             if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#             current_qpos = np.array(current_qpos, dtype=np.float32)
#         norm_qpos = (current_qpos - self.action_mean) / self.action_std
        
#         # for _ in range(self.history_len):
#         #     self.state_buffer.append(norm_qpos)
#         self.state_buffer.append(norm_qpos)

#     # @torch.no_grad()
#     # def step(self, frames_list, current_qpos):
#     #     """
#     #     :param frames_list: 包含若干帧真实历史图像的列表 (通常是客户端发来的最新几帧)
#     #     """
#     #     # 1. 更新 Video Buffer
#     #     # 注意：客户端可能发来 16 帧，也可能只发来最新 1 帧。
#     #     # 我们将它们全部 append 到长 Buffer 中。
#     #     for frame in frames_list:
#     #         frame_resized = cv2.resize(frame, (224, 224))
#     #         frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     #         wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
#     #         main_fake = torch.zeros_like(wrist_tensor)
#     #         combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
#     #         self.video_buffer.append(combined_frame)
        
#     #     # 2. State Preprocess & Update
#     #     if len(current_qpos) == 7:
#     #         current_qpos = list(current_qpos) + [0.0]
        
#     #     qpos_np = np.array(current_qpos, dtype=np.float32)
#     #     norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
#     #     # 更新状态 Buffer (只存最新的即可，或者存历史)
#     #     # 这里简单起见，append 最新的
#     #     self.state_buffer.append(norm_qpos_np)
        
#     #     # =========================================================
#     #     # 🟢 核心：均匀采样 (Uniform Sampling)
#     #     # =========================================================
#     #     curr_len = len(self.video_buffer)
#     #     # 从 Buffer 中均匀选取 model_input_frames (6) 帧
#     #     # np.linspace 生成均匀间隔的索引
#     #     indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
        
#     #     # 取出选中的帧
#     #     buffer_list = list(self.video_buffer)
#     #     selected_frames = [buffer_list[i] for i in indices]
        
#     #     # 堆叠 -> [6, 2, 3, 224, 224]
#     #     vid_t = torch.stack(selected_frames).to(self.device)
#     #     # 调整维度 -> [1, 2, 3, 6, 224, 224] (Batch=1, T=6)
#     #     vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)
        

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         # 1. 更新 Video Buffer
#         for frame in frames_list:
#             wrist_tensor = self.preprocess_image(frame)
#             main_fake = torch.zeros_like(wrist_tensor)
#             combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
#             self.video_buffer.append(combined_frame) 
#             # 队列会自动挤出旧的，保持最新的500帧
        
#         # 2. 更新 State
#         if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#         qpos_np = np.array(current_qpos, dtype=np.float32)
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos_np)
        
#         # 🟢 [修改] 动态均匀采样 (核心逻辑)
#         curr_len = len(self.video_buffer)
        
#         # 无论当前 Buffer 是 1 帧还是 500 帧，都均匀取出 6 帧
#         # 这保证了模型始终能看到“全历史”的概貌，而不是“局部静止切片”
#         indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
        
#         selected_frames = [self.video_buffer[i] for i in indices]
        
#         # Stack -> [6, 2, 3, 224, 224]
#         vid_t = torch.stack(selected_frames).to(self.device)
#         # Permute -> [1, 2, 3, 6, 224, 224] (Batch, View, Channel, Time, H, W)
#         vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)

#         # State: 取当前最新的状态即可 (因为 FusionEncoder 只用 state[:, -1, :])
#         # 为了格式统一，我们构造一个 [1, 1, 8] 的 Tensor
#         state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
#         # 4. Inference
#         self.scheduler.set_timesteps(self.inference_steps)
#         with autocast('cuda', dtype=torch.bfloat16):
#             # (1) 获取视觉特征 (包含 ForeSight 的未来预测)
#             features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            
#             # (2) 手动注入当前 State (确保 RDT 拿到的是最新本体感知)
#             # state_t 是 [1, 1, 8], 取 [:, -1, :] 得到 [1, 8]
#             features["state"] = state_t[:, -1, :] 
            
#             latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#             for t in self.scheduler.timesteps:
#                 model_input = self.scheduler.scale_model_input(latents, t)
#                 t_tensor = torch.tensor([t], device=self.device)
                
#                 noise_pred = self.policy(model_input, t_tensor, features)
                
#                 latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
#         normalized_actions = latents[0].float()
#         action_pred_np = normalized_actions.detach().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         # =========================================================
#         # 🚨 [关键修复] 夹爪二值化 (Thresholding)
#         # =========================================================
#         # 定义物理极限
#         GRIPPER_OPEN_VAL = 0.0804  
#         GRIPPER_CLOSE_VAL = 0.0428 
#         GRIPPER_THRESHOLD = 0.0616 

#         # 获取原始预测值
#         raw_gripper_pred = denormalized_actions[:, 7]

#         # 二值化判断
#         binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
        
#         # 覆盖回去
#         denormalized_actions[:, 7] = binary_gripper
        
#         print(f"   >>> [Gripper] Raw: {raw_gripper_pred[0]:.4f} -> Binary: {binary_gripper[0]:.4f}", end='\r')
#         # =========================================================
        
#         safe_actions = self.safety.clip_actions(denormalized_actions)
#         return safe_actions.tolist()


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
import torch._dynamo
from torchvision import transforms

# === 导入你的模型 ===
from model.fusion_encoder import FusionEncoder
from model.rdt_model import RDTWrapper

# === 基础路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = "/yanghaochuan/data/115dataset_stats.json" 
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
STAGE_C_PATH = '/yanghaochuan/114checkpoints_finetune/StageC_ForeSight_step_10000.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafetyController:
    def __init__(self):
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
        
        import time
        import os
        # 定义保存目录
        self.debug_dir = f"debug_visuals_{int(time.time())}"
        os.makedirs(self.debug_dir, exist_ok=True)
        self.step_counter = 0

        print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        except:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
        # 加载统计数据
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        
        mean_raw = np.array(stats['action_mean'], dtype=np.float32)
        std_raw = np.array(stats['action_std'], dtype=np.float32)
        
        if mean_raw.shape[0] > 8:
            self.action_mean = mean_raw[:8]
            self.action_std = std_raw[:8]
        elif mean_raw.shape[0] == 7:
            self.action_mean = np.concatenate([mean_raw, [0.0]])
            self.action_std = np.concatenate([std_raw, [1.0]])
        else:
            self.action_mean = mean_raw
            self.action_std = std_raw
            
        self.action_std = np.maximum(self.action_std, 1e-2)

        print(f"📊 [Stats Loaded] Mean[0]: {self.action_mean[0]:.3f}, GripperMean: {self.action_mean[7]:.3f}")
        print(f"📊 [Stats Loaded] Std[0]:  {self.action_std[0]:.3f}, GripperStd:  {self.action_std[7]:.3f}")

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self._init_models()
        self._init_scheduler()
        
        self.video_buffer = deque(maxlen=self.history_len)
        self.state_buffer = deque(maxlen=self.history_len)
        self.first_frame_tensor = None
        self.text_tokens = None 
        self.default_prompt = "pick up the orange ball and put it on the plank"
        
        # 🟢 [诊断] 关闭 torch.compile 以排除编译错误干扰
        # torch._dynamo.config.suppress_errors = True
        # try: self.encoder = torch.compile(self.encoder, mode="default")
        # except: pass
        self.warmup()

    def _init_models(self):
        print(f"[Agent] Initializing models on {self.device}...")
        self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
        self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
        print(f"[Agent] Loading Checkpoint: {STAGE_C_PATH}")
        ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
        self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
        if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
        else: self.policy.load_state_dict(ckpt_c, strict=False)
        
        if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)

    def _init_scheduler(self):
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
        self.inference_steps = 25
        self.scheduler.set_timesteps(self.inference_steps)

    def warmup(self):
        print("🔥 [System] Warming up model...")
        dummy_video = torch.randn(1, 2, 3, 6, 224, 224, device=self.device, dtype=torch.bfloat16)
        dummy_text = torch.randint(0, 1000, (1, 16), device=self.device)
        dummy_state = torch.randn(1, 1, 8, device=self.device, dtype=torch.float32)
        dummy_ff = torch.randn(1, 2, 3, 224, 224, device=self.device, dtype=torch.float32)
        try:
            with autocast('cuda', dtype=torch.bfloat16):
                feats = self.encoder(dummy_video, dummy_text, dummy_state, dummy_ff)
                feats["state"] = dummy_state[:, -1, :]
                latents = torch.randn(1, self.pred_horizon, 8, device=self.device)
                t = torch.tensor([0], device=self.device)
                _ = self.policy(latents, t, feats)
            print("✅ Warmup done.")
        except Exception as e:
            print(f"❌ Warmup failed: {e}")

    def preprocess_image(self, img_np):
        resized = cv2.resize(img_np, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor = self.normalize(tensor) 
        return tensor

    def save_debug_image(self, tensor, name="debug.png"):
        try:
            t = tensor.detach().cpu().clone()
            # Un-Normalize: x * std + mean
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = t * std + mean
            t = torch.clamp(t, 0, 1)
            img_np = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(name, img_bgr)
            # print(f"📸 [Debug] Saved model input view to {name}")
        except Exception as e:
            pass

    # 🟢 [新增] 这是一个专门把 Tensor 还原成图片的函数
    def save_model_input_visuals(self, vid_tensor, step_idx):
        """
        将模型输入的 6 帧 Tensor 反归一化并拼图保存
        vid_tensor shape: [1, 2, 3, 6, 224, 224] (Batch, View, Channel, Time, H, W)
        """
        try:
            # 取出 wrist 视角 (View Index 1), 去掉 Batch 维 -> [3, 6, 224, 224]
            # 注意：你的代码里 Main 是 0 (全黑), Wrist 是 1
            wrist_t = vid_tensor[0, 1] 
            
            # 反归一化参数 (ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).to(wrist_t.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).to(wrist_t.device)
            
            # 反归一化: x * std + mean
            wrist_t = wrist_t * std + mean
            wrist_t = torch.clamp(wrist_t, 0, 1)
            
            # 转为 Numpy: [3, 6, 224, 224] -> [6, 224, 224, 3]
            imgs = wrist_t.permute(1, 2, 3, 0).detach().cpu().numpy()
            imgs = (imgs * 255).astype(np.uint8)
            
            # 拼接 6 帧成一行长图
            # imgs[0] 是 Buffer 里最早的一帧，imgs[-1] 是最新的一帧
            concat_img = np.hstack([imgs[i] for i in range(6)])
            
            # 转为 BGR 供 cv2 保存
            concat_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
            
            # 保存
            save_path = os.path.join(self.debug_dir, f"step_{step_idx:04d}_buffer.jpg")
            cv2.imwrite(save_path, concat_img)
            # print(f"📸 Saved buffer visual to {save_path}") # 刷屏可注释掉
            
        except Exception as e:
            print(f"⚠️ Visualization Failed: {e}")

    def reset_session(self, first_frame_img, current_qpos=None):
        print("[Agent] Resetting session (Cold Start)...")
        self.video_buffer.clear()
        self.state_buffer.clear()
        
        # === 🟢 [Double Check] 处理首帧 ===
        wrist_tensor = self.preprocess_image(first_frame_img)
        main_fake = torch.zeros_like(wrist_tensor)
        self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        self.save_debug_image(wrist_tensor, "debug_first_frame_wrist.png")
        
        tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
        self.text_tokens = tokens.to(self.device)
        
        video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
        self.video_buffer.append(video_frame_unit)
            
        if current_qpos is None: current_qpos = np.zeros(8)
        else: 
            if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
            current_qpos = np.array(current_qpos, dtype=np.float32)
            
        # 🔍 打印初始状态
        print(f"   🚩 [Reset QPos] {current_qpos[:6]} ... Grip: {current_qpos[7]}")
        
        norm_qpos = (current_qpos - self.action_mean) / self.action_std
        self.state_buffer.append(norm_qpos)

    @torch.no_grad()
    def step(self, frames_list, current_qpos):
        # 1. 更新 Video
        for frame in frames_list:
            wrist_tensor = self.preprocess_image(frame)
            main_fake = torch.zeros_like(wrist_tensor)
            combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
            self.video_buffer.append(combined_frame) 
        
        # 2. 更新 State
        if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
        qpos_np = np.array(current_qpos, dtype=np.float32)
        
        # 🟢 [核心诊断] 计算 Normalized State
        norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        self.state_buffer.append(norm_qpos_np)
        
        # =================================================================
        # 🚨 [关键监控] 如果这里的数值 > 3.0 或 < -3.0，说明状态输入错了！
        # =================================================================
        gripper_norm = norm_qpos_np[7]
        joint0_norm = norm_qpos_np[0]
        if abs(gripper_norm) > 3.0 or abs(joint0_norm) > 3.0:
             print(f"\n⚠️ STATE OOD! J0_Norm: {joint0_norm:.2f}, Grip_Norm: {gripper_norm:.2f} | Raw Grip: {qpos_np[7]:.4f}")
        
        # 3. 采样
        curr_len = len(self.video_buffer)
        indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
        selected_frames = [self.video_buffer[i] for i in indices]
        
        vid_t = torch.stack(selected_frames).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)

        # =========================================================
        # 🟢 [插入] 在这里保存模型看到的画面！
        # =========================================================
        self.save_model_input_visuals(vid_t, self.step_counter)
        self.step_counter += 1

        # State: 取当前 state
        state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 4. Inference
        self.scheduler.set_timesteps(self.inference_steps)
        with autocast('cuda', dtype=torch.bfloat16):
            features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            features["state"] = state_t[:, -1, :] 
            latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
            for t in self.scheduler.timesteps:
                model_input = self.scheduler.scale_model_input(latents, t)
                t_tensor = torch.tensor([t], device=self.device)
                noise_pred = self.policy(model_input, t_tensor, features)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        normalized_actions = latents[0].float()
        # === 🟢 [新增] 诊断代码 ===
        # 计算当前预测动作的“平均绝对值”
        mean_abs_val = torch.mean(torch.abs(normalized_actions)).item()
        
        # 打印第一步的归一化数值 (看它是不是全是 0.x)
        first_step_norm = normalized_actions[0].detach().cpu().numpy()
        print(f"\n🔍 [Diagnosis] Normalized Mean Abs: {mean_abs_val:.4f}")
        print(f"   First Step Norm: {np.round(first_step_norm, 3)}")
        # =========================
        action_pred_np = normalized_actions.detach().cpu().numpy()
        denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
        # 夹爪二值化
        GRIPPER_OPEN_VAL = 0.0804  
        GRIPPER_CLOSE_VAL = 0.0428 
        GRIPPER_THRESHOLD = 0.0616 

        raw_gripper_pred = denormalized_actions[:, 7]
        binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
        denormalized_actions[:, 7] = binary_gripper
        
        print(f"   >>> [Step] NormState J0: {joint0_norm:.2f} G: {gripper_norm:.2f} | Pred J0: {denormalized_actions[0,0]:.3f}", end='\r')
        
        safe_actions = self.safety.clip_actions(denormalized_actions)
        return safe_actions.tolist()