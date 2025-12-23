import torch
import cv2
import json
import time
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

# === 基础路径 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = "/yanghaochuan/projects/data/dataset_stats.json"
TOKENIZER_PATH = "/yanghaochuan/models/flan-t5-large"

STAGE_B_PATH = '/yanghaochuan/projects/checkpoints/stageB_papercup.pt'
STAGE_C_PATH = '/yanghaochuan/projects/checkpoints/stageC_lora_epoch_40.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# ⚡️ 核心参数调优 (稳健版)
# =============================================================================
# 1. 增益回归正常 (防止乱动)
EXECUTION_GAIN = 1.0 

# 2. 【大招】Z轴重力偏置 (Gravity Bias)
# 只要夹爪是开的，每一步都额外向下压 4cm。这是打破悬停的神器。
Z_AXIS_BIAS = -0.04 

# =============================================================================
# 🛡️ 安全控制器
# =============================================================================
class SafetyController:
    def __init__(self, joint_dim=7):
        self.joint_limits_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]) + 0.05
        self.joint_limits_max = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89]) - 0.05
        self.max_delta = 0.06
        self.last_action = None

    def apply_safety(self, target_qpos, current_qpos):
        target_qpos = np.array(target_qpos)
        current_qpos = np.array(current_qpos)
        
        target_joints = target_qpos[:7]
        current_joints = current_qpos[:7]
        target_gripper = target_qpos[7:] 

        delta = target_joints - current_joints
        delta_clipped = np.clip(delta, -self.max_delta, self.max_delta)
        
        safe_joints = current_joints + delta_clipped
        safe_joints = np.clip(safe_joints, self.joint_limits_min, self.joint_limits_max)

        safe_action_8d = np.concatenate([safe_joints, target_gripper])
        
        self.last_action = safe_action_8d
        return safe_action_8d.tolist()

    def reset(self):
        self.last_action = None


# =============================================================================
# 🤖 实时推理 Agent
# =============================================================================
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
        self.action_mean = np.array(stats['action_mean'], dtype=np.float32)
        self.action_std = np.array(stats['action_std'], dtype=np.float32)
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
        print(f"[Agent] 📉 Gravity Bias Enabled: {Z_AXIS_BIAS}m per step")

    def _init_models(self):
        print(f"[Agent] Initializing models on {self.device}...")
        try:
            self.encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(self.device).eval()
            self.policy = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768).to(self.device).eval()
            
            print(f"[Agent] Loading Encoder: {STAGE_B_PATH}")
            ckpt_b = torch.load(STAGE_B_PATH, map_location=self.device)
            if isinstance(ckpt_b, dict) and 'encoder_state_dict' in ckpt_b:
                self.encoder.load_state_dict(ckpt_b['encoder_state_dict'], strict=False)
            else:
                self.encoder.load_state_dict(ckpt_b, strict=False)

            print(f"[Agent] Loading Policy: {STAGE_C_PATH}")
            peft_config = LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
                lora_dropout=0.05, bias="none"
            )
            self.policy.rdt_model = get_peft_model(self.policy.rdt_model, peft_config)
            
            ckpt_c = torch.load(STAGE_C_PATH, map_location=self.device)
            if 'rdt_state_dict' in ckpt_c:
                self.policy.load_state_dict(ckpt_c['rdt_state_dict'], strict=False)
            else:
                self.policy.load_state_dict(ckpt_c, strict=False)
            print("[Agent] ✅ All weights loaded.")

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
        self.inference_steps = 10
        self.scheduler.set_timesteps(self.inference_steps)

    def reset_session(self, first_frame_img):
        print("[Agent] Resetting session...")
        self.video_buffer.clear()
        self.state_buffer.clear()
        self.safety.reset() 
        
        # 1. 处理手腕图像 (Wrist Image)
        # 调整大小 -> RGB -> Tensor [3, 224, 224]
        ff_resized = cv2.resize(first_frame_img, (224, 224))
        ff_rgb = cv2.cvtColor(ff_resized, cv2.COLOR_BGR2RGB)
        wrist_tensor = torch.tensor(ff_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 2. 构造全黑主摄 (Fake Main Image) - 关键修改
        # 这一步是为了匹配训练时的 Modality Dropout 和 step 函数的输入格式
        main_fake = torch.zeros_like(wrist_tensor)
        
        # 3. 堆叠得到双摄帧 (Stack Views)
        # Shape: [2, 3, 224, 224] (View 0: Fake Main, View 1: Real Wrist)
        dual_frame = torch.stack([main_fake, wrist_tensor], dim=0)
        
        # 4. 设置 First Frame Tensor (Task Anchor)
        # 增加 Batch 维度 -> [1, 2, 3, 224, 224]
        self.first_frame_tensor = dual_frame.unsqueeze(0).to(self.device)
        
        # 5. 编码文本指令 (Text Instruction)
        tokens = self.tokenizer(
            self.default_prompt, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=16, 
            truncation=True
        ).input_ids
        self.text_tokens = tokens.to(self.device)
        
        # 6. 预填充缓冲区 (Buffer Padding)
        # 使用第一帧重复填充，防止 step 函数因为数据不足而报错
        # 注意：这里存入的是 dual_frame (Tensor)，而不是原始图片
        for _ in range(self.window_size):
            self.video_buffer.append(dual_frame) 
            self.state_buffer.append(np.zeros(8)) # 假设归一化后的"均值状态"为0


    # def reset_session(self, first_frame_img):
    #     print("[Agent] Resetting session...")
    #     self.video_buffer.clear()
    #     self.state_buffer.clear()
    #     self.safety.reset() 
        
    #     ff = cv2.resize(first_frame_img, (224, 224))
    #     ff = cv2.cvtColor(ff, cv2.COLOR_BGR2RGB)
    #     tensor = torch.tensor(ff, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        
    #     self.first_frame_tensor = tensor.to(self.device) / 255.0
        
    #     tokens = self.tokenizer(
    #         self.default_prompt, 
    #         return_tensors="pt", 
    #         padding="max_length", 
    #         max_length=16, 
    #         truncation=True
    #     ).input_ids
    #     self.text_tokens = tokens.to(self.device)
        
    #     for _ in range(self.window_size):
    #         self.video_buffer.append(ff)
    #         self.state_buffer.append(np.zeros(8)) 


    @torch.no_grad()
    def step(self, current_frame, current_qpos):
        """
        输入:
            current_frame: numpy array, [H, W, 3], BGR (cv2 default), 手腕相机图像
            current_qpos: list or array, 长度 7 (关节) 或 8 (关节+夹爪)
        输出:
            actions: list of lists, shape [16, 8], 包含未来16步的动作序列
        """
        # 1. 预处理手腕图像 (Wrist Image)
        frame_resized = cv2.resize(current_frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # [H, W, 3] -> [3, H, W] -> Normalize
        wrist_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 2. 构造全黑的主摄图像 (Fake Main Image)
        # 这一步至关重要：它模拟了训练时的 Dropout 情况，让模型以为"主摄丢了"，从而自信地使用手腕相机
        main_fake = torch.zeros_like(wrist_tensor)
        
        # 3. 堆叠双摄 (Stack Views)
        # Result shape: [2, 3, 224, 224] (View 0: Fake Main, View 1: Real Wrist)
        combined_frame = torch.stack([main_fake, wrist_tensor], dim=0)
        
        # 4. 存入缓冲区 (Video Buffer)
        self.video_buffer.append(combined_frame)
        
        # 5. 处理状态 (Proprioception)
        if len(current_qpos) == 7:
            current_qpos = list(current_qpos) + [0.0] # 补齐夹爪
            
        # 归一化当前状态
        # 注意：这里必须使用与 Dataset Loader 相同的统计量 (Z-Score)
        raw_qpos_t = torch.tensor(current_qpos, dtype=torch.float32, device=self.device)
        norm_qpos_t = (raw_qpos_t - self.action_mean) / self.action_std
        self.state_buffer.append(norm_qpos_t.cpu().numpy()) # 存入 buffer
        
        # 6. 冷启动处理 (Cold Start)
        # 如果历史数据不足 window_size (16帧)，则无法推理，返回当前位置保持不动
        if len(self.video_buffer) < self.window_size:
            # 返回 16 步"保持当前位置"的动作
            return [current_qpos] * self.pred_horizon
        
        # 7. 构造模型输入 Tensor
        # Video Buffer: list of [2, 3, H, W] (len=16) -> Stack -> [16, 2, 3, H, W]
        # Permute to: [Batch, Views, Channels, Time, Height, Width]
        # -> [1, 2, 3, 16, 224, 224]
        vid_t = torch.stack(list(self.video_buffer)).to(self.device)
        vid_t = vid_t.permute(1, 2, 0, 3, 4).unsqueeze(0)
        
        # State Buffer: [1, 16, 8]
        state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # First Frame (Task Anchor)
        # 注意：reset_session 时也应该构造类似的双摄 First Frame
        # 这里假设 self.first_frame_tensor 已经是 [1, 2, 3, H, W] 格式
        # 如果 reset_session 还没改，请确保那里也加上 main_fake 逻辑
        ff_t = self.first_frame_tensor 

        # 8. 模型推理 (Inference)
        self.scheduler.set_timesteps(self.inference_steps)
        
        with autocast('cuda', dtype=torch.bfloat16):
            # A. 编码特征 (VideoMAE + Fusion)
            # Encoder 内部会自动处理双摄特征融合
            features = self.encoder(vid_t, self.text_tokens, state_t, ff_t)
            
            # B. 扩散去噪 (Diffusion Sampling)
            # Latents 初始化: [Batch, Horizon, Action_Dim] -> [1, 16, 8]
            # 这就是 Action Chunking 的核心：一次生成 16 步
            latents = torch.randn(1, self.pred_horizon, 8, device=self.device) 
            
            for t in self.scheduler.timesteps:
                model_input = self.scheduler.scale_model_input(latents, t)
                t_tensor = torch.tensor([t], device=self.device)
                
                # RDT 预测噪声
                noise_pred = self.policy(model_input, t_tensor, features)
                
                # 移除噪声
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 9. 后处理与反归一化
        # [1, 16, 8] -> [16, 8]
        normalized_actions = latents[0].float()
        
        # 反归一化: (Norm * Std) + Mean
        denormalized_actions = (normalized_actions * self.action_std + self.action_mean).cpu().numpy()
        
        # 10. 返回动作序列 Chunk
        return denormalized_actions.tolist()

    # @torch.no_grad()
    # def step(self, current_frame, current_qpos):
    #     frame_resized = cv2.resize(current_frame, (224, 224))
    #     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
    #     if len(current_qpos) == 7:
    #         current_qpos = list(current_qpos) + [0.0]
        
    #     raw_qpos = np.array(current_qpos, dtype=np.float32)
    #     norm_qpos = (raw_qpos - self.action_mean) / self.action_std
        
    #     self.video_buffer.append(frame_rgb)
    #     self.state_buffer.append(norm_qpos) 
        
    #     if len(self.video_buffer) < self.window_size:
    #         return current_qpos 
        
    #     vid_t = torch.tensor(np.array(list(self.video_buffer)), dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(self.device) / 255.0
    #     state_t = torch.tensor(np.array(list(self.state_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
        
    #     self.scheduler.set_timesteps(self.inference_steps)
        
    #     with autocast('cuda', dtype=torch.bfloat16):
    #         features = self.encoder(vid_t, self.text_tokens, state_t, self.first_frame_tensor)
    #         latents = torch.randn(1, 1, 8, device=self.device) 
    #         for t in self.scheduler.timesteps:
    #             model_input = self.scheduler.scale_model_input(latents, t)
    #             t_tensor = torch.tensor([t], device=self.device)
    #             noise_pred = self.policy(model_input, t_tensor, features)
    #             latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
    #     normalized_action = latents[0, 0].float().cpu().numpy()
    #     model_target = normalized_action * self.action_std + self.action_mean
        
    #     # === 核心修正逻辑 ===
        
    #     # 1. 恢复正常增益 (1.0)，消除震荡
    #     final_target_qpos = raw_qpos + (model_target - raw_qpos) * EXECUTION_GAIN
        
    #     # 2. 【Gravity Bias】如果夹爪是开的 (还没抓到)，强制向下压
    #     gripper_is_open = current_qpos[7] > 0.04 
    #     if gripper_is_open:

    #         pass
            
    #     # 修正：既然是 Joint Space，我们稍微放大一点点 Gain，但绝不能 2.5
    #     final_target_qpos = raw_qpos + (model_target - raw_qpos) * 1.2
        
    #     safe_action = self.safety.apply_safety(target_qpos=final_target_qpos, current_qpos=current_qpos)
    #     return safe_action

# =============================================================================
# 服务器主循环
# =============================================================================
def run_safe_server(host='0.0.0.0', port=6000):
    # 这个函数主要是给 server_gpu_image.py 调用的
    # 或者如果单独运行此脚本进行测试，可以用下面的逻辑
    pass 

if __name__ == '__main__':
    # 为了方便测试，你可以在这里把 server_gpu_image.py 的逻辑粘过来
    # 但通常是直接运行 python inference/server_gpu_image.py
    pass