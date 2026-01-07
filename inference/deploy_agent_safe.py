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

# inference/deploy_agent_safe.py
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

# === 导入你的模型 ===
from model.fusion_encoder import FusionEncoder
from model.rdt_model import RDTWrapper

# === 基础路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = "/yanghaochuan/data/16dataset_stats.json" # 确保这里读取的是新生成的 stats
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
STAGE_C_PATH = '/yanghaochuan/16checkpoints_finetune/12stageC_step_3800.pt' # 记得改成你新训练的 checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafetyController:
    def __init__(self):
        # Franka 关节极限 (安全余量)
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
        
        # 维度修正
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
        
        self._init_models()
        self._init_scheduler()
        
        self.window_size = 16
        self.video_buffer = deque(maxlen=self.window_size)
        self.state_buffer = deque(maxlen=self.window_size)
        self.first_frame_tensor = None
        self.text_tokens = None 
        self.default_prompt = "pick up the orange ball"

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
        
        torch._dynamo.config.suppress_errors = True
        try: self.encoder = torch.compile(self.encoder, mode="default")
        except: pass

    def _init_scheduler(self):
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True)
        self.inference_steps = 25
        self.scheduler.set_timesteps(self.inference_steps)

    def reset_session(self, first_frame_img, current_qpos=None):
        print("[Agent] Resetting session (Cold Start)...")
        self.video_buffer.clear()
        self.state_buffer.clear()
        
        ff_resized = cv2.resize(first_frame_img, (224, 224))
        ff_rgb = cv2.cvtColor(ff_resized, cv2.COLOR_BGR2RGB)
        wrist_tensor = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        main_fake = torch.zeros_like(wrist_tensor)
        self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
        tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
        self.text_tokens = tokens.to(self.device)
        
        # 填满 buffer (冷启动)
        video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0) 
        for _ in range(self.window_size):
            self.video_buffer.append(video_frame_unit) 
            
        if current_qpos is None: current_qpos = np.zeros(8)
        else: 
            if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
            current_qpos = np.array(current_qpos, dtype=np.float32)
        norm_qpos = (current_qpos - self.action_mean) / self.action_std
        for _ in range(self.window_size):
            self.state_buffer.append(norm_qpos)

    @torch.no_grad()
    def step(self, frames_list, current_qpos):
        """
        :param frames_list: 包含 16 帧真实历史图像的列表 (List[np.array])
        """
        # 1. 刷新 Video Buffer (填入真实的 16 帧历史)
        self.video_buffer.clear()
        for frame in frames_list:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            main_fake = torch.zeros_like(wrist_tensor)
            combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
            self.video_buffer.append(combined_frame)
        
        while len(self.video_buffer) < self.window_size:
            self.video_buffer.append(self.video_buffer[-1])

        # 2. State Preprocess
        if len(current_qpos) == 7:
            current_qpos = list(current_qpos) + [0.0]
        
        qpos_np = np.array(current_qpos, dtype=np.float32)
        norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
        self.state_buffer.clear()
        for _ in range(self.window_size):
            self.state_buffer.append(norm_qpos_np)
        
        # 3. Batch Construction
        vid_t = torch.stack(list(self.video_buffer)).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0) 
        state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Inference
        self.scheduler.set_timesteps(self.inference_steps)
        with autocast('cuda', dtype=torch.bfloat16):
            features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            # 手动注入 State
            features["state"] = state_t[:, -1, :] 
            
            latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            for t in self.scheduler.timesteps:
                model_input = self.scheduler.scale_model_input(latents, t)
                t_tensor = torch.tensor([t], device=self.device)
                noise_pred = self.policy(model_input, t_tensor, features)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        normalized_actions = latents[0].float()
        action_pred_np = normalized_actions.detach().cpu().numpy()
        denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
        # =========================================================
        # 🚨 [关键修复] 夹爪二值化 (Thresholding)
        # =========================================================
        # 假设夹爪在第 8 维 (index 7)
        # 计算该维度的物理中点 (基于你之前生成的 Stats)
        gripper_midpoint = self.action_mean[7] 
        
        # 读取当前预测的夹爪值
        raw_gripper_pred = denormalized_actions[:, 7]
        
        # 1. 定义物理极限 (直接填你跑出来的数值)
        GRIPPER_OPEN_VAL = 0.0804  # 张开
        GRIPPER_CLOSE_VAL = 0.0428 # 闭合 (或者稍微小一点 0.04 以确保抓紧)
        GRIPPER_THRESHOLD = 0.0616 # 阈值

        # 2. 获取模型预测的原始物理值
        raw_gripper_pred = denormalized_actions[:, 7]

        # 3. 二值化判断
        # 大于阈值 -> 设为 Open
        # 小于阈值 -> 设为 Close
        binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
        
        # 4. 覆盖回去
        denormalized_actions[:, 7] = binary_gripper
        
        print(f"   >>> [Gripper] Raw: {raw_gripper_pred[0]:.4f} -> Binary: {binary_gripper[0]:.4f}", end='\r')
        # =========================================================
        
        safe_actions = self.safety.clip_actions(denormalized_actions)
        return safe_actions.tolist()