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
# STATS_PATH = "/yanghaochuan/data/115dataset_stats.json" 
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# STAGE_C_PATH = '/yanghaochuan/114checkpoints_finetune/StageC_ForeSight_step_10000.pt'

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
#         self.history_len = 500       
#         self.model_input_frames = 6 
        
#         import time
#         import os
#         # 定义保存目录
#         self.debug_dir = f"debug_visuals_{int(time.time())}"
#         os.makedirs(self.debug_dir, exist_ok=True)
#         self.step_counter = 0

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

#         print(f"📊 [Stats Loaded] Mean[0]: {self.action_mean[0]:.3f}, GripperMean: {self.action_mean[7]:.3f}")
#         print(f"📊 [Stats Loaded] Std[0]:  {self.action_std[0]:.3f}, GripperStd:  {self.action_std[7]:.3f}")

#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                               std=[0.229, 0.224, 0.225])
#         self._init_models()
#         self._init_scheduler()
        
#         self.video_buffer = deque(maxlen=self.history_len)
#         self.state_buffer = deque(maxlen=self.history_len)
#         self.first_frame_tensor = None
#         self.text_tokens = None 
#         self.default_prompt = "pick up the orange ball and put it on the plank"
        
#         # 🟢 [诊断] 关闭 torch.compile 以排除编译错误干扰
#         # torch._dynamo.config.suppress_errors = True
#         # try: self.encoder = torch.compile(self.encoder, mode="default")
#         # except: pass
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
#         tensor = self.normalize(tensor) 
#         return tensor

#     def save_debug_image(self, tensor, name="debug.png"):
#         try:
#             t = tensor.detach().cpu().clone()
#             # Un-Normalize: x * std + mean
#             mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#             std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#             t = t * std + mean
#             t = torch.clamp(t, 0, 1)
#             img_np = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#             img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#             cv2.imwrite(name, img_bgr)
#             # print(f"📸 [Debug] Saved model input view to {name}")
#         except Exception as e:
#             pass

#     # 🟢 [新增] 这是一个专门把 Tensor 还原成图片的函数
#     def save_model_input_visuals(self, vid_tensor, step_idx):
#         """
#         将模型输入的 6 帧 Tensor 反归一化并拼图保存
#         vid_tensor shape: [1, 2, 3, 6, 224, 224] (Batch, View, Channel, Time, H, W)
#         """
#         try:
#             # 取出 wrist 视角 (View Index 1), 去掉 Batch 维 -> [3, 6, 224, 224]
#             # 注意：你的代码里 Main 是 0 (全黑), Wrist 是 1
#             wrist_t = vid_tensor[0, 1] 
            
#             # 反归一化参数 (ImageNet)
#             mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).to(wrist_t.device)
#             std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).to(wrist_t.device)
            
#             # 反归一化: x * std + mean
#             wrist_t = wrist_t * std + mean
#             wrist_t = torch.clamp(wrist_t, 0, 1)
            
#             # 转为 Numpy: [3, 6, 224, 224] -> [6, 224, 224, 3]
#             imgs = wrist_t.permute(1, 2, 3, 0).detach().cpu().numpy()
#             imgs = (imgs * 255).astype(np.uint8)
            
#             # 拼接 6 帧成一行长图
#             # imgs[0] 是 Buffer 里最早的一帧，imgs[-1] 是最新的一帧
#             concat_img = np.hstack([imgs[i] for i in range(6)])
            
#             # 转为 BGR 供 cv2 保存
#             concat_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
            
#             # 保存
#             save_path = os.path.join(self.debug_dir, f"step_{step_idx:04d}_buffer.jpg")
#             cv2.imwrite(save_path, concat_img)
#             # print(f"📸 Saved buffer visual to {save_path}") # 刷屏可注释掉
            
#         except Exception as e:
#             print(f"⚠️ Visualization Failed: {e}")

#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session (Cold Start)...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         # === 🟢 [Double Check] 处理首帧 ===
#         wrist_tensor = self.preprocess_image(first_frame_img)
#         main_fake = torch.zeros_like(wrist_tensor)
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
#         self.save_debug_image(wrist_tensor, "debug_first_frame_wrist.png")
        
#         tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
#         self.text_tokens = tokens.to(self.device)
        
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
#         self.video_buffer.append(video_frame_unit)
            
#         if current_qpos is None: current_qpos = np.zeros(8)
#         else: 
#             if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#             current_qpos = np.array(current_qpos, dtype=np.float32)
            
#         # 🔍 打印初始状态
#         print(f"   🚩 [Reset QPos] {current_qpos[:6]} ... Grip: {current_qpos[7]}")
        
#         norm_qpos = (current_qpos - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos)

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         # 1. 更新 Video
#         for frame in frames_list:
#             wrist_tensor = self.preprocess_image(frame)
#             main_fake = torch.zeros_like(wrist_tensor)
#             combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
#             self.video_buffer.append(combined_frame) 
        
#         # 2. 更新 State
#         if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#         qpos_np = np.array(current_qpos, dtype=np.float32)
        
#         # 🟢 [核心诊断] 计算 Normalized State
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
#         self.state_buffer.append(norm_qpos_np)
        
#         # =================================================================
#         # 🚨 [关键监控] 如果这里的数值 > 3.0 或 < -3.0，说明状态输入错了！
#         # =================================================================
#         gripper_norm = norm_qpos_np[7]
#         joint0_norm = norm_qpos_np[0]
#         if abs(gripper_norm) > 3.0 or abs(joint0_norm) > 3.0:
#              print(f"\n⚠️ STATE OOD! J0_Norm: {joint0_norm:.2f}, Grip_Norm: {gripper_norm:.2f} | Raw Grip: {qpos_np[7]:.4f}")
        
#         # 3. 采样
#         curr_len = len(self.video_buffer)
#         indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
#         selected_frames = [self.video_buffer[i] for i in indices]
        
#         vid_t = torch.stack(selected_frames).to(self.device)
#         vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)

#         # =========================================================
#         # 🟢 [插入] 在这里保存模型看到的画面！
#         # =========================================================
#         self.save_model_input_visuals(vid_t, self.step_counter)
#         self.step_counter += 1

#         # State: 取当前 state
#         state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
#         # 4. Inference
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
            
#         normalized_actions = latents[0].float()
#         # === 🟢 [新增] 诊断代码 ===
#         # 计算当前预测动作的“平均绝对值”
#         mean_abs_val = torch.mean(torch.abs(normalized_actions)).item()
        
#         # 打印第一步的归一化数值 (看它是不是全是 0.x)
#         first_step_norm = normalized_actions[0].detach().cpu().numpy()
#         print(f"\n🔍 [Diagnosis] Normalized Mean Abs: {mean_abs_val:.4f}")
#         print(f"   First Step Norm: {np.round(first_step_norm, 3)}")
#         # =========================
#         action_pred_np = normalized_actions.detach().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
#         # 夹爪二值化
#         GRIPPER_OPEN_VAL = 0.0804  
#         GRIPPER_CLOSE_VAL = 0.0428 
#         GRIPPER_THRESHOLD = 0.0616 

#         raw_gripper_pred = denormalized_actions[:, 7]
#         binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
#         denormalized_actions[:, 7] = binary_gripper
        
#         print(f"   >>> [Step] NormState J0: {joint0_norm:.2f} G: {gripper_norm:.2f} | Pred J0: {denormalized_actions[0,0]:.3f}", end='\r')
        
#         safe_actions = self.safety.clip_actions(denormalized_actions)
#         return safe_actions.tolist()




# ego单视角
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
import time

# === 导入你的模型 ===
from model.fusion_encoder import FusionEncoder
from model.rdt_model import RDTWrapper

# === 基础路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = "/yanghaochuan/data/124dataset_stats.json" 
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
STAGE_C_PATH = '/yanghaochuan/124checkpoints_finetune/StageC_ForeSight_step_7000.pt'

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
        self.trajectory_offset = None
        
        # 🟢 [Alignment] 与 dataset_loader.py 保持一致
        self.history_len = 500       # 模拟 dataset 中的 history_len
        self.model_input_frames = 6  # 模拟 dataset 中的 window_size
        
        self.debug_dir = f"debug_visuals_{int(time.time())}"
        os.makedirs(self.debug_dir, exist_ok=True)
        self.step_counter = 0

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

        # 🟢 [Alignment] 归一化参数与 VideoMAE/Dataset 一致
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self._init_models()
        self._init_scheduler()
        
        # 🟢 [Alignment] 历史 Buffer，对应 Dataset 中的 sliding window
        self.video_buffer = deque(maxlen=self.history_len)
        self.state_buffer = deque(maxlen=self.history_len)
        
        self.first_frame_tensor = None
        self.text_tokens = None 
        self.default_prompt = "pick up the orange ball and put it on the plank"
        
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
        
        # if 'encoder_state_dict' in ckpt_c: self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
        if 'encoder_state_dict' in ckpt_c: 
            print("正在加载 Encoder 权重...")
            state_dict = ckpt_c['encoder_state_dict']
            
            # 🛠️ 修复：移除编译或DDP产生的前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                k_clean = k.replace("_orig_mod.", "").replace("module.", "")
                new_state_dict[k_clean] = v
            
            # 🔍 诊断：不要用 strict=False，或者打印返回值
            missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
            
            if len(missing) > 0:
                print(f"⚠️ 警告：Encoder 加载有丢失键! (数量: {len(missing)})")
                print(f"   示例丢失: {missing[:5]}")
            else:
                print("✅ Encoder 权重完美加载！")

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
        # 🟢 [Alignment] 必须归一化
        tensor = self.normalize(tensor) 
        return tensor

    def save_model_input_visuals(self, vid_tensor, step_idx):
        try:
            wrist_t = vid_tensor[0, 1] 
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1).to(wrist_t.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1).to(wrist_t.device)
            wrist_t = wrist_t * std + mean
            wrist_t = torch.clamp(wrist_t, 0, 1)
            imgs = wrist_t.permute(1, 2, 3, 0).detach().cpu().numpy()
            imgs = (imgs * 255).astype(np.uint8)
            concat_img = np.hstack([imgs[i] for i in range(6)])
            concat_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(self.debug_dir, f"step_{step_idx:04d}_buffer.jpg")
            cv2.imwrite(save_path, concat_img)
        except Exception as e:
            print(f"⚠️ Visualization Failed: {e}")

    def reset_session(self, first_frame_img, current_qpos=None):
        print("[Agent] Resetting session (Cold Start)...")
        self.video_buffer.clear()
        self.state_buffer.clear()
        
        # 🟢 [Alignment] 首帧处理
        wrist_tensor = self.preprocess_image(first_frame_img)
        main_fake = torch.zeros_like(wrist_tensor)
        self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
        tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
        self.text_tokens = tokens.to(self.device)
        
        # Buffer 初始填入这一帧
        video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
        self.video_buffer.append(video_frame_unit)
            
        if current_qpos is None: current_qpos = np.zeros(8)
        else: 
            if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
            current_qpos = np.array(current_qpos, dtype=np.float32)
            
        print(f"   🚩 [Reset QPos] {current_qpos[:7]} ... Grip: {current_qpos[7]}")
        
        norm_qpos = (current_qpos - self.action_mean) / self.action_std
        self.state_buffer.append(norm_qpos)
        # === 🟢 添加这几行诊断代码 ===
        print(f"\n🔍 [Stats Check] J0 Mean: {self.action_mean[0]:.4f}, Std: {self.action_std[0]:.4f}")
        print(f"📉 [Input Norm Check] Current J0: {current_qpos[0]:.4f} -> Normalized: {norm_qpos[0]:.4f}")
        if abs(norm_qpos[0]) > 3.0:
            print("⚠️ 警告：初始状态严重偏离训练分布 (OOD)！模型可能会失效！")
        # ============================
        self.trajectory_offset = None  # 新增：确保每次新动作开始时重新计算对齐
        print("[Agent] Trajectory offset reset.")

    @torch.no_grad()
    def step(self, frames_list, current_qpos):
        """
        Stop-and-Think 模式:
        1. 接收 frames_list (这些是机器人在执行上一个动作片段时捕获的‘历史’帧)
        2. 将它们**全部**加入 Buffer (模拟时间流逝)
        3. 进行均匀采样 (模拟 Training Loader)
        4. 推理下一个动作
        """
        # ========================================================
        # 🟢 Phase 1: Update History (Movement Phase Replay)
        # ========================================================
        # 将传入的所有帧按顺序加入 Buffer
        # 这完全对应了训练集中，滑窗随着时间步 t 前进而前进
        for frame in frames_list:
            wrist_tensor = self.preprocess_image(frame)
            main_fake = torch.zeros_like(wrist_tensor)
            combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
            self.video_buffer.append(combined_frame) 
        
        # 🟢 State Update
        # 我们假设这批图像对应的状态近似于当前状态 (或者你可以让Client传状态列表)
        # 为了保证 Video/State Buffer 长度对齐，我们重复 append 当前状态
        if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
        qpos_np = np.array(current_qpos, dtype=np.float32)
        norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
        # 重复填充，使得状态历史长度与视觉历史长度匹配 (虽然模型只用最后一个)
        for _ in range(len(frames_list)):
            self.state_buffer.append(norm_qpos_np)
        
        # ========================================================
        # 🟢 Phase 2: Inference (Stop Phase)
        # ========================================================
        
        # 1. Uniform Sampling (完全复刻 Dataset __getitem__ 逻辑)
        curr_len = len(self.video_buffer)
        
        # np.linspace(0, valid_len-1, 6)
        indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
        selected_frames = [self.video_buffer[i] for i in indices]
        
        # 构造 Batch
        vid_t = torch.stack(selected_frames).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0) # [1, 2, 3, 6, H, W]

        # 保存 Debug 图片 (确认模型到底看到了什么)
        self.save_model_input_visuals(vid_t, self.step_counter)
        self.step_counter += 1

        # State: 取最新的 (FusionEncoder 只关注当前时刻)
        state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 2. Diffusion Inference
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
            
        # 3. Denormalize & Output
        normalized_actions = latents[0].float()
        action_pred_np = normalized_actions.detach().cpu().numpy()
        denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
        # 夹爪二值化
        GRIPPER_OPEN_VAL = 0.0804  
        GRIPPER_CLOSE_VAL = 0.0428 
        GRIPPER_THRESHOLD = 0.0616 

        raw_gripper_pred = denormalized_actions[:, 7]
        binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
        denormalized_actions[:, 7] = binary_gripper

        if self.trajectory_offset is None:
            # 计算模型预测的第 0 步与当前机器人真实位置的差值
            # 只针对前 7 个关节 (J0-J6)
            pred_start = denormalized_actions[0, :7]
            real_start = qpos_np[:7]
            self.trajectory_offset = pred_start - real_start
            print(f"🚩 [Aligner] Offset calculated: {self.trajectory_offset}")
            
        # === 将打印逻辑移到这里 ===
        print(f"\n{'='*25} ALIGNED RDT Action (First 15 Steps) {'='*25}")
        header = f"{'Step':<4} | {'J0':^7} {'J1':^7} {'J2':^7} {'J3':^7} {'J4':^7} {'J5':^7} {'J6':^7} | {'Grip':^6}"
        print(header)
        for i in range(min(15, len(denormalized_actions))):
            step_data = denormalized_actions[i]
            joints_str = " ".join([f"{x: .4f}" for x in step_data[:7]])
            print(f"{i:<4} | {joints_str} | {step_data[7]:.4f}")
        # ========================


        # 1. 获取实时位置 (qpos_np 是你在 step 开始时处理好的当前物理状态)
        real_start_pos = qpos_np[:8] 

        # 2. 强制覆盖 Step 0，确保物理层面绝对重合
        # 这样机器人执行第一个动作时就不会有任何“瞬跳”
        denormalized_actions[0, :8] = real_start_pos
        
        # 简单日志
        print(f"   >>> [Infer] BufferLen: {curr_len} | Pred J0: {denormalized_actions[0,0]:.3f}", end='\r')
        
        safe_actions = self.safety.clip_actions(denormalized_actions)
        return safe_actions.tolist()


# #ego双视角
# import torch
# import cv2
# import json
# import numpy as np
# from collections import deque
# from diffusers import DDIMScheduler
# import os
# import h5py
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
# STATS_PATH = "/yanghaochuan/data/121dataset_stats.json" 
# TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"
# STAGE_C_PATH = '/yanghaochuan/121checkpoints_finetune/StageC_ForeSight_step_7000.pt'
# ANCHOR_DATA_PATH = '/yanghaochuan/data/hdf5/pick_up_the_orange_ball_and_put_it_on_the_plank.hdf5'

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
#         self.history_len = 500       
#         self.model_input_frames = 6 
        
#         self.debug_dir = f"debug_visuals_{int(time.time())}"
#         os.makedirs(self.debug_dir, exist_ok=True)
#         self.step_counter = 0

#         # === 1. 初始化对齐器变量 ===
#         self.trajectory_offset = None # 用于存储 (Model_Start - Real_Start) 的偏差

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

#         # 保留之前的统计学补丁，防止数值爆炸
#         PATCH_STD_VAL = 0.5
#         if self.action_std[3] < PATCH_STD_VAL: self.action_std[3] = PATCH_STD_VAL
#         if self.action_std[5] < PATCH_STD_VAL: self.action_std[5] = PATCH_STD_VAL

#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         self.anchor_main_tensor = self._load_anchor_image()

#         self._init_models()
#         self._init_scheduler()
        
#         self.video_buffer = deque(maxlen=self.history_len)
#         self.state_buffer = deque(maxlen=self.history_len)
#         self.first_frame_tensor = None
#         self.text_tokens = None 
#         self.default_prompt = "pick up the orange ball and put it on the plank"
        
#         self.warmup()

#     def _load_anchor_image(self):
#         print(f"📥 [Agent] Loading Anchor Image from {ANCHOR_DATA_PATH}...")
#         try:
#             with h5py.File(ANCHOR_DATA_PATH, 'r') as f:
#                 demo_grp = f['data']['demo_0'] 
#                 main_key = 'agentview_image' if 'agentview_image' in demo_grp['obs'] else 'agentview_rgb'
#                 img_np = demo_grp['obs'][main_key][0]
#                 cv2.imwrite("debug_anchor_main.jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
#                 return self.preprocess_image(img_np)
#         except Exception as e:
#             print(f"⚠️ Anchor Load Failed: {e}")
#             return None

#     def _init_models(self):
#         print(f"[Agent] Initializing models on {self.device}...")
#         self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
#         self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
        
#         ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
#         peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], lora_dropout=0.05, bias="none")
#         self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
        
#         if 'rdt_state_dict' in ckpt_c: self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
#         else: self.policy.load_state_dict(ckpt_c, strict=False)
        
#         if 'encoder_state_dict' in ckpt_c: 
#             state_dict = ckpt_c['encoder_state_dict']
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 k_clean = k.replace("_orig_mod.", "").replace("module.", "")
#                 new_state_dict[k_clean] = v
#             self.encoder.load_state_dict(new_state_dict, strict=False)

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
#         except Exception: pass

#     def reset_session(self, first_frame_img, current_qpos=None):
#         print("[Agent] Resetting session (Cold Start)...")
#         self.video_buffer.clear()
#         self.state_buffer.clear()
        
#         # 重置对齐偏差，这将在第一次 step 时重新计算
#         self.trajectory_offset = None
        
#         wrist_tensor = self.preprocess_image(first_frame_img)
        
#         if self.anchor_main_tensor is not None:
#             main_fake = self.anchor_main_tensor.clone()
#         else:
#             main_fake = torch.zeros_like(wrist_tensor)
            
#         self.first_frame_tensor = torch.stack([main_fake, wrist_tensor], dim=0).unsqueeze(0).to(self.device)
        
#         tokens = self.tokenizer(self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True).input_ids
#         self.text_tokens = tokens.to(self.device)
        
#         video_frame_unit = torch.stack([main_fake, wrist_tensor], dim=0)
            
#         if current_qpos is None: current_qpos = np.zeros(8)
#         else: 
#             if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#             current_qpos = np.array(current_qpos, dtype=np.float32)
            
#         norm_qpos = (current_qpos - self.action_mean) / self.action_std
        
#         for _ in range(self.model_input_frames):
#             self.video_buffer.append(video_frame_unit)
#             self.state_buffer.append(norm_qpos)
            
#         print(f"   🚩 [Reset QPos] {current_qpos[:6]}")

#     @torch.no_grad()
#     def step(self, frames_list, current_qpos):
#         # 1. 更新 Buffer
#         for frame in frames_list:
#             wrist_tensor = self.preprocess_image(frame)
#             if self.anchor_main_tensor is not None:
#                 main_fake = self.anchor_main_tensor.clone()
#             else:
#                 main_fake = torch.zeros_like(wrist_tensor)
            
#             combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
#             self.video_buffer.append(combined_frame) 
        
#         if len(current_qpos) == 7: current_qpos = list(current_qpos) + [0.0]
#         qpos_np = np.array(current_qpos, dtype=np.float32)
#         norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
#         for _ in range(len(frames_list)):
#             self.state_buffer.append(norm_qpos_np)
        
#         # 2. 准备输入
#         curr_len = len(self.video_buffer)
#         indices = np.linspace(0, curr_len - 1, self.model_input_frames).astype(int)
#         selected_frames = [self.video_buffer[i] for i in indices]
        
#         vid_t = torch.stack(selected_frames).to(self.device)
#         vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0) 

#         self.save_model_input_visuals(vid_t, self.step_counter)
#         self.step_counter += 1

#         state_t = torch.tensor(norm_qpos_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
#         # 3. 模型推理
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
            
#         normalized_actions = latents[0].float()
#         action_pred_np = normalized_actions.detach().cpu().numpy()
#         denormalized_actions = action_pred_np * self.action_std + self.action_mean

#         # =========================================================
#         # 🟢 [核心修复] 轨迹对齐器 (Trajectory Aligner)
#         # =========================================================
#         # 逻辑：在 Reset 后的第一步，计算 Model 认为的起始点与真实起始点的差值 (Offset)
#         # 然后将这个 Offset 应用到整条轨迹，强制 Step 0 == Real Position
#         # =========================================================
        
#         if self.trajectory_offset is None:
#             # 1. 获取模型预测的 Step 0 (只看前 7 个关节，不改夹爪)
#             pred_start = denormalized_actions[0, :7]
#             real_start = qpos_np[:7]
            
#             # 2. 计算偏差: Model - Real
#             self.trajectory_offset = pred_start - real_start
            
#             print(f"\n   🔧 [Aligner] Calibration Done.")
#             print(f"      Real Start:  {real_start[3]:.3f} (J3), {real_start[5]:.3f} (J5)")
#             print(f"      Model Start: {pred_start[3]:.3f} (J3), {pred_start[5]:.3f} (J5)")
#             print(f"      Offset:      {self.trajectory_offset[3]:.3f} (J3), {self.trajectory_offset[5]:.3f} (J5)")
#             print(f"      >> Applying negative offset to align trajectory.\n")

#         # 应用对齐 (Subtract Offset)
#         # 只对关节应用，不对夹爪应用
#         denormalized_actions[:, :7] -= self.trajectory_offset
        
#         # =========================================================

#         # 夹爪二值化
#         GRIPPER_OPEN_VAL = 0.0804  
#         GRIPPER_CLOSE_VAL = 0.0428 
#         GRIPPER_THRESHOLD = 0.0616 

#         raw_gripper_pred = denormalized_actions[:, 7]
#         binary_gripper = np.where(raw_gripper_pred > GRIPPER_THRESHOLD, GRIPPER_OPEN_VAL, GRIPPER_CLOSE_VAL)
#         denormalized_actions[:, 7] = binary_gripper
        
#         print(f"   >>> [Infer] BufferLen: {curr_len} | Pred J0: {denormalized_actions[0,0]:.3f} | J3: {denormalized_actions[0,3]:.3f} | J5: {denormalized_actions[0,5]:.3f}", end='\r')
        
#         safe_actions = self.safety.clip_actions(denormalized_actions)
#         return safe_actions.tolist()