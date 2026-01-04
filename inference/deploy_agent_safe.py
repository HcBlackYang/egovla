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

# # [修改点] 指向 Stage C 的 checkpoint
# STAGE_C_PATH = '/yanghaochuan/checkpoints/12stageC_step_4000.pt'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =============================================================================
# # 🛡️ 安全控制器
# # =============================================================================
# class SafetyController:
#     def __init__(self):
#         # Franka 关节极限 (安全余量 0.05)
#         self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.05
#         self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.05

#     def clip_actions(self, actions_batch):
#         actions_np = np.array(actions_batch)
#         joints = actions_np[:, :7]
#         gripper = actions_np[:, 7:]
#         # 关节限位
#         joints_clipped = np.clip(joints, self.joint_limits_min, self.joint_limits_max)
#         return np.concatenate([joints_clipped, gripper], axis=1)

# # =============================================================================
# # 🤖 实时推理 Agent
# # =============================================================================
# class RealTimeAgent:
#     def __init__(self):
#         self.device = DEVICE
#         self.safety = SafetyController() 
#         self.pred_horizon = 64  # 明确模型预测长度

#         print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
#         try:
#             self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
#         except:
#             self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
#         # --- 1. 加载统计数据并修正维度 ---
#         if not os.path.exists(STATS_PATH):
#             raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
#         with open(STATS_PATH, 'r') as f:
#             stats = json.load(f)
        
#         mean_raw = np.array(stats['action_mean'], dtype=np.float32)
#         std_raw = np.array(stats['action_std'], dtype=np.float32)
        
#         # 强制截断或补齐到 8 维 (7 Joint + 1 Gripper)
#         if mean_raw.shape[0] > 8:
#             print(f"[Agent] ⚠️ 统计数据维度 {mean_raw.shape[0]} > 8，进行截断。")
#             self.action_mean = mean_raw[:8]
#             self.action_std = std_raw[:8]
#         elif mean_raw.shape[0] == 7:
#             print(f"[Agent] ⚠️ 统计数据维度为 7，补齐 Gripper=0。")
#             self.action_mean = np.concatenate([mean_raw, [0.0]])
#             self.action_std = np.concatenate([std_raw, [1.0]])
#         else:
#             self.action_mean = mean_raw
#             self.action_std = std_raw
            
#         self.action_std = np.maximum(self.action_std, 1e-2)
        
#         self._init_models()
#         self._init_scheduler()
        
#         self.window_size = 16
#         # 使用 deque 自动维护滑动窗口
#         self.video_buffer = deque(maxlen=self.window_size)
#         self.state_buffer = deque(maxlen=self.window_size)
        
#         self.first_frame_tensor = None
#         self.text_tokens = None 
        
#         self.default_prompt = "pick up the orange ball"
#         print(f"[Agent] Prompt: '{self.default_prompt}'")

#     def _init_models(self):
#         print(f"[Agent] Initializing models on {self.device}...")
#         try:
#             # Init Base Models
#             self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
#             self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
            
#             # Load Stage C Checkpoint
#             print(f"[Agent] 🚀 Loading Joint Checkpoint: {STAGE_C_PATH}")
#             ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)

#             # Load Policy (LoRA)
#             peft_config = LoraConfig(
#                 r=16, lora_alpha=32,
#                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
#                 lora_dropout=0.05, bias="none"
#             )
#             self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
            
#             if 'rdt_state_dict' in ckpt_c:
#                 self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
#             else:
#                 self.policy.load_state_dict(ckpt_c, strict=False)
#             print("[Agent] ✅ Policy weights loaded.")

#             # Load Encoder (Joint Finetuned)
#             if 'encoder_state_dict' in ckpt_c:
#                 self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
#                 print("[Agent] ✅ Encoder weights loaded from Stage C.")
#             else:
#                 raise ValueError(f"❌ 严重错误: {STAGE_C_PATH} 中没有 'encoder_state_dict'！")
            
#             # Compile (Optional)
#             print("[Agent] Compiling FusionEncoder...")
#             torch._dynamo.config.suppress_errors = True
#             try:
#                 self.encoder = torch.compile(self.encoder, mode="default")
#             except Exception as e:
#                 print(f"[Warning] Encoder compile failed: {e}")

#         except Exception as e:
#             print(f"[Error] Model Init Failed: {e}")
#             raise e

#     def _init_scheduler(self):
#         self.scheduler = DDIMScheduler(
#             num_train_timesteps=1000,
#             beta_schedule="squaredcos_cap_v2",
#             prediction_type="epsilon", 
#             clip_sample=True
#         )
#         self.inference_steps = 25
#         self.scheduler.set_timesteps(self.inference_steps)

#     def reset_session(self, first_frame_img, current_qpos=None):
#         """
#         重置会话，实现冷启动逻辑
#         :param first_frame_img: 全局首帧（RGB，HWC）
#         :param current_qpos: 初始机械臂状态（可选，7或8维）
#         """
#         print("[Agent] Resetting session (Cold Start)...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         # 1. 处理首帧 (Context) - 这是我们的锚点
#         ff_resized = cv2.resize(first_frame_img, (224, 224))
#         ff_rgb = cv2.cvtColor(ff_resized, cv2.COLOR_BGR2RGB)
#         wrist_tensor = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
#         # 构造全黑主摄 (Inference Mode: Main is always black)
#         main_fake = torch.zeros_like(wrist_tensor)
        
#         # Context Frame: [2, 3, 224, 224]
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         # 2. 编码指令
#         tokens = self.tokenizer(
#             self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True
#         ).input_ids
#         self.text_tokens = tokens.to(self.device)
        
#         # 3. 预填充 Video Buffer (冷启动：复制首帧 16 次)
#         # 注意：这里假设初始时刻机器人视角也近似于首帧（或者我们没有更好的历史）
#         # 实际推理中，Main View 是全黑的，Wrist View 是当前的 RGB
#         # 这里为了简便，Video Buffer 里的 Main 也是全黑，Wrist 是首帧图像
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0) # [2, 3, H, W]
        
#         for _ in range(self.window_size):
#             self.video_buffer.append(video_frame_unit) 
            
#         # 4. 预填充 State Buffer
#         if current_qpos is None:
#             current_qpos = np.zeros(8)
#         else:
#              if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#              current_qpos = np.array(current_qpos, dtype=np.float32)

#         # 归一化初始状态
#         norm_qpos = (current_qpos - self.action_mean) / self.action_std
#         for _ in range(self.window_size):
#             self.state_buffer.append(norm_qpos)

#     @torch.no_grad()
#     def step(self, current_frame, current_qpos):
#         """
#         单步推理
#         :param current_frame: 当前手腕相机图像 (RGB)
#         :param current_qpos: 当前机械臂关节状态
#         """
#         # 1. Image Preprocess (Wrist View)
#         frame_resized = cv2.resize(current_frame, (224, 224))
#         frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#         wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
#         # Main View is ALWAYS Black during inference (Student Mode)
#         main_fake = torch.zeros_like(wrist_tensor)
#         combined_frame = torch.stack([main_fake, wrist_tensor], dim=0) # [2, 3, 224, 224]
        
#         # =========================================================================
#         # [核心修复] 滑动窗口策略 (Sliding Window Strategy)
#         # 不要 clear()! 只是 append，deque 会自动挤出最旧的一帧。
#         # 这样保留了时序动态信息。
#         # =========================================================================
#         self.video_buffer.append(combined_frame)
        
#         # 2. State Preprocess
#         if len(current_qpos) == 7:
#             current_qpos = list(current_qpos) + [0.0]
        
#         qpos_np = np.array(current_qpos, dtype=np.float32)
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
#         self.state_buffer.append(norm_qpos_np)
        
#         # 3. Batch Construction
#         # Video: [1, 2, 3, 16, 224, 224]
#         # list(deque) 会按时序返回列表
#         vid_t = torch.stack(list(self.video_buffer)).to(self.device)
#         vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)
        
#         # State: [1, 16, 8]
#         state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         # 4. Inference
#         self.scheduler.set_timesteps(self.inference_steps)
        
#         with autocast('cuda', dtype=torch.bfloat16):
#             # Encoder 接收: 
#             # - vid_t: 当前滑窗视频 (Main黑, Wrist实)
#             # - first_frame_tensor: 完美的全局首帧 (Context Anchor)
#             features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            
#             # RDT 预测未来 64 步
#             latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
#             for t in self.scheduler.timesteps:
#                 model_input = self.scheduler.scale_model_input(latents, t)
#                 t_tensor = torch.tensor([t], device=self.device)
#                 noise_pred = self.policy(model_input, t_tensor, features)
#                 latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
#         # 5. Post-process
#         # 我们通常只取第一步或前几步动作执行 (Receding Horizon Control)
#         # 这里返回完整的 64 步，由 robot_policy_system 决定执行多少步
#         normalized_actions = latents[0].float()
        
#         action_pred_np = normalized_actions.detach().cpu().numpy() # [64, 8]
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         # 安全限位
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

# === 导入你的模型 ===
from model.fusion_encoder import FusionEncoder
from model.rdt_model import RDTWrapper

# === 基础路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = "/yanghaochuan/data/13dataset_stats.json"
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
STAGE_C_PATH = '/yanghaochuan/checkpoints/12stageC_step_4000.pt'

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

        print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        except:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
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
        
        self._init_models()
        self._init_scheduler()
        
        self.window_size = 16
        self.video_buffer = deque(maxlen=self.window_size)
        self.state_buffer = deque(maxlen=self.window_size)
        self.first_frame_tensor = None
        self.text_tokens = None 
        self.default_prompt = "pick up the orange ball"

    def _init_models(self):
        # ... (模型初始化代码保持不变) ...
        print(f"[Agent] Initializing models on {self.device}...")
        self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
        self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
        self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
        if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
        else: self.policy.load_state_dict(ckpt_c, strict=False)
        
        if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
        else: raise ValueError(f"❌ No encoder_state_dict in {STAGE_C_PATH}")
        
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
        
        # Buffer 初始化 (虽然马上会被 step 覆盖，但为了安全先填满)
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
        :param current_qpos: 当前机器人关节状态
        """
        # =========================================================================
        # [逻辑修正] 完全重置 Buffer，填入真实的 16 帧历史
        # =========================================================================
        self.video_buffer.clear()
        
        for frame in frames_list:
            # 预处理每一帧
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            main_fake = torch.zeros_like(wrist_tensor)
            combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
            
            self.video_buffer.append(combined_frame)
        
        # 确保填满了 (如果客户端传来的不足16帧，应该在客户端补齐，但这里双重保险)
        while len(self.video_buffer) < self.window_size:
            self.video_buffer.append(self.video_buffer[-1])

        # 2. State Preprocess
        if len(current_qpos) == 7:
            current_qpos = list(current_qpos) + [0.0]
        
        qpos_np = np.array(current_qpos, dtype=np.float32)
        norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
        # 状态 Buffer 也应该刷新，但通常我们只有当前状态
        # 策略：假设过去16帧的状态都近似于当前状态 (或者你可以让客户端也传状态历史)
        # 这里简化处理：填满当前状态
        self.state_buffer.clear()
        for _ in range(self.window_size):
            self.state_buffer.append(norm_qpos_np)
        
        # 3. Batch Construction
        vid_t = torch.stack(list(self.video_buffer)).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0) # [1, 2, 3, 16, 224, 224]
        
        state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Inference
        self.scheduler.set_timesteps(self.inference_steps)
        # with autocast('cuda', dtype=torch.bfloat16):
        #     features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
        #     latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
        #     for t in self.scheduler.timesteps:
        #         model_input = self.scheduler.scale_model_input(latents, t)
        #         t_tensor = torch.tensor([t], device=self.device)
        #         noise_pred = self.policy(model_input, t_tensor, features)
        #         latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        with autocast('cuda', dtype=torch.bfloat16):
            # (1) 获取视觉特征
            features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            
            # =================================================================
            # 🚨 [关键修复] 手动注入 State！
            # 必须与训练时的 behavior 一致：取时间窗口的最后一帧 state[:, -1, :]
            # =================================================================
            features["state"] = state_t[:, -1, :] 
            
            latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
            for t in self.scheduler.timesteps:
                model_input = self.scheduler.scale_model_input(latents, t)
                t_tensor = torch.tensor([t], device=self.device)
                
                # (2) 传入包含 state 的完整字典
                noise_pred = self.policy(model_input, t_tensor, features)
                
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # ... (后续代码不变) ...
            
        normalized_actions = latents[0].float()
        action_pred_np = normalized_actions.detach().cpu().numpy()
        denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
        gripper_val = denormalized_actions[0, 7]
        print(f"   >>> [Model Output] Gripper: {gripper_val:.4f} (Threshold: <0.06 Close)", end='\r')
        safe_actions = self.safety.clip_actions(denormalized_actions)
        return safe_actions.tolist()