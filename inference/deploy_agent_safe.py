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
STATS_PATH = "/yanghaochuan/data/1223dataset_stats.json"
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"

# [修改点] 不再需要 Stage B 路径，直接用 Stage C
STAGE_C_PATH = '/yanghaochuan/checkpoints/checkpoint_step_3200.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 🛡️ 安全控制器
# =============================================================================
class SafetyController:
    def __init__(self):
        # Franka 关节极限 (安全余量 0.05)
        self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.05
        self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.05

    def clip_actions(self, actions_batch):
        actions_np = np.array(actions_batch)
        joints = actions_np[:, :7]
        gripper = actions_np[:, 7:]
        # 关节限位
        joints_clipped = np.clip(joints, self.joint_limits_min, self.joint_limits_max)
        return np.concatenate([joints_clipped, gripper], axis=1)

# =============================================================================
# 🤖 实时推理 Agent
# =============================================================================
class RealTimeAgent:
    def __init__(self):
        self.device = DEVICE
        self.safety = SafetyController() 
        self.pred_horizon = 64  # 明确模型预测长度

        print(f"[Agent] Loading Tokenizer from {TOKENIZER_PATH}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        except:
            self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
        # --- 1. 加载统计数据并修正维度 ---
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"❌ 找不到统计文件: {STATS_PATH}")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        
        mean_raw = np.array(stats['action_mean'], dtype=np.float32)
        std_raw = np.array(stats['action_std'], dtype=np.float32)
        
        # 强制截断或补齐到 8 维 (7 Joint + 1 Gripper)
        if mean_raw.shape[0] > 8:
            print(f"[Agent] ⚠️ 统计数据维度 {mean_raw.shape[0]} > 8，进行截断。")
            self.action_mean = mean_raw[:8]
            self.action_std = std_raw[:8]
        elif mean_raw.shape[0] == 7:
            print(f"[Agent] ⚠️ 统计数据维度为 7，补齐 Gripper=0。")
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
        
        self.default_prompt = "pick up the paper cup"
        print(f"[Agent] Prompt: '{self.default_prompt}'")

    def _init_models(self):
        print(f"[Agent] Initializing models on {self.device}...")
        try:
            # Init Base Models
            self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
            self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768, pred_horizon=64).to(self.device).eval()
            
            # Load Stage C Checkpoint
            print(f"[Agent] 🚀 Loading Joint Checkpoint: {STAGE_C_PATH}")
            ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)

            # Load Policy (LoRA)
            peft_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
                lora_dropout=0.05, bias="none"
            )
            self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
            
            if 'rdt_state_dict' in ckpt_c:
                self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
            else:
                self.policy.load_state_dict(ckpt_c, strict=False)
            print("[Agent] ✅ Policy weights loaded.")

            # Load Encoder (Joint Finetuned)
            if 'encoder_state_dict' in ckpt_c:
                self.encoder.load_state_dict(ckpt_c['encoder_state_dict'], strict=False)
                print("[Agent] ✅ Encoder weights loaded from Stage C.")
            else:
                raise ValueError(f"❌ 严重错误: {STAGE_C_PATH} 中没有 'encoder_state_dict'！")
            
            # Compile (Optional)
            print("[Agent] Compiling FusionEncoder...")
            torch._dynamo.config.suppress_errors = True
            try:
                self.encoder = torch.compile(self.encoder, mode="default")
            except Exception as e:
                print(f"[Warning] Encoder compile failed: {e}")

        except Exception as e:
            print(f"[Error] Model Init Failed: {e}")
            raise e

    def _init_scheduler(self):
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon", 
            clip_sample=True
        )
        self.inference_steps = 25
        self.scheduler.set_timesteps(self.inference_steps)

    def reset_session(self, first_frame_img):
        print("[Agent] Resetting session...")
        self.video_buffer.clear()
        self.state_buffer.clear()
        
        # 处理首帧
        ff_resized = cv2.resize(first_frame_img, (224, 224))
        ff_rgb = cv2.cvtColor(ff_resized, cv2.COLOR_BGR2RGB)
        wrist_tensor = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 构造全黑主摄
        main_fake = torch.zeros_like(wrist_tensor)
        dual_frame = torch.stack([main_fake, wrist_tensor], dim=0)
        
        self.first_frame_tensor = dual_frame.unsqueeze(0).to(self.device)
        
        # 编码指令
        tokens = self.tokenizer(
            self.default_prompt, return_tensors="pt", padding="max_length", max_length=16, truncation=True
        ).input_ids
        self.text_tokens = tokens.to(self.device)
        
        # 预填充 Buffer
        for _ in range(self.window_size):
            self.video_buffer.append(dual_frame) 
            self.state_buffer.append(np.zeros(8)) 

    @torch.no_grad()
    def step(self, current_frame, current_qpos):
        # 1. Image Preprocess
        frame_resized = cv2.resize(current_frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        main_fake = torch.zeros_like(wrist_tensor)
        combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
        
        # =========================================================================
        # [核心修复] 静态图策略 (Static Image Strategy)
        # 每次推理前，清空 Buffer，用【当前帧】填满它。
        # 解决低频推理导致的 "视频时序错乱" 问题。
        # =========================================================================
        self.video_buffer.clear()
        for _ in range(self.window_size):
            self.video_buffer.append(combined_frame)
        
        # 2. State Preprocess
        if len(current_qpos) == 7:
            current_qpos = list(current_qpos) + [0.0]
        
        # [Safety Fix] 使用 Numpy 在 CPU 上计算，不要创建 GPU Tensor
        qpos_np = np.array(current_qpos, dtype=np.float32)
        norm_qpos_np = (qpos_np - self.action_mean) / self.action_std
        
        # 同样清空 State Buffer 并填满
        self.state_buffer.clear()
        for _ in range(self.window_size):
            self.state_buffer.append(norm_qpos_np)
        
        # 3. Batch Construction
        # Video: [1, 2, 3, 16, 224, 224]
        vid_t = torch.stack(list(self.video_buffer)).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)
        
        # State: [1, 16, 8] - 这里才把 NumPy 转 Tensor 并移到 GPU
        state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Inference
        self.scheduler.set_timesteps(self.inference_steps)
        
        with autocast('cuda', dtype=torch.bfloat16):
            features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
            latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
            for t in self.scheduler.timesteps:
                model_input = self.scheduler.scale_model_input(latents, t)
                t_tensor = torch.tensor([t], device=self.device)
                noise_pred = self.policy(model_input, t_tensor, features)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 5. Post-process
        normalized_actions = latents[0].float()
        
        # [Safety Fix] 先转 CPU Numpy，再进行反归一化运算
        action_pred_np = normalized_actions.detach().cpu().numpy() # [B, 8]
        denormalized_actions = action_pred_np * self.action_std + self.action_mean
        
        # 安全限位
        safe_actions = self.safety.clip_actions(denormalized_actions)
        return safe_actions.tolist()