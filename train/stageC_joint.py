

# # train/stageC_joint.py
# import sys
# import os
# import torch
# import torch.optim as optim
# import argparse
# import time
# from torch.utils.data import DataLoader
# from torch.amp import autocast
# from diffusers import DDPMScheduler
# # === 新增：导入 PEFT ===
# from peft import LoraConfig, get_peft_model, TaskType

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.fusion_encoder import FusionEncoder
# from models.rdt_model import RDTWrapper
# from utils.dataset_loader import RobotDataset

# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# PRED_HORIZON = 16

# def train_stage_c(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Stage C Training: RDT Policy (LoRA) on {device} ===")
    
#     # 1. 模型初始化
#     print("Loading FusionEncoder...")
#     fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
#     if os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B Weights: {args.stage_b_ckpt}")
#         fusion_encoder.load_state_dict(torch.load(args.stage_b_ckpt), strict=False)
#     else:
#         print(f"Warning: Stage B Checkpoint not found at {args.stage_b_ckpt}!")
    
#     print("Freezing FusionEncoder...")
#     fusion_encoder.eval()
#     for param in fusion_encoder.parameters():
#         param.requires_grad = False
    
#     print("Loading RDT Policy...")
#     rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768).to(device)
    
#     # === 核心修改：应用 LoRA ===
#     print("Applying LoRA to RDT...")
#     # 针对 Transformer 的常见 Linear 层名称
#     # 如果报错找不到层，可以打印 rdt_wrapper.rdt_model 看看具体层名
#     peft_config = LoraConfig(
#         r=16,               # Rank
#         lora_alpha=32,      # Alpha
#         target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
#         lora_dropout=0.05,
#         bias="none",
#         # task_type="FEATURE_EXTRACTION" # RDT 本质是去噪预测
#     )
    
#     # 我们只对 RDT 内部的核心模型加 LoRA，而不是 Wrapper 壳子
#     # 这里的 rdt_wrapper.rdt_model 是实际的 Transformer
#     try:
#         rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
#         rdt_wrapper.rdt_model.print_trainable_parameters()
#     except Exception as e:
#         print(f"LoRA 自动挂载失败 (可能是层名不匹配): {e}")
#         print("尝试全量微调 (不推荐)...")
#         # 如果自动挂载失败，就fallback到普通训练
    
#     # 2. 调度器与优化器
#     noise_scheduler = DDPMScheduler(
#         num_train_timesteps=1000,
#         beta_schedule="squaredcos_cap_v2",
#         prediction_type="sample"
#     )
    
#     # 只优化 LoRA 参数 (requires_grad=True 的参数)
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), lr=1e-4, weight_decay=1e-4)

#     # 3. 数据加载 (会自动读取 stats.json)
#     print(f"Loading data from: {args.data_root}")
#     dataset = RobotDataset(hdf5_path=args.data_root, window_size=16)
    
#     loader = DataLoader(
#         dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         num_workers=args.num_workers,
#         pin_memory=True,
#         persistent_workers=True,
#         drop_last=True
#     )

#     # 4. 训练循环
#     rdt_wrapper.train()
#     print(">>> LoRA Training Started... <<<")
    
#     for epoch in range(args.epochs):
#         total_loss = 0
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             video = batch['video'].to(device, non_blocking=True)
#             state = batch['state'].to(device, non_blocking=True) 
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
#             actions = batch['action_target'].to(device, non_blocking=True)

#             if torch.rand(1).item() < 0.5:
#                 # 这种写法要求 FusionEncoder 内部能处理 0 输入
#                 # 或者在这里直接把 tensor 归零
#                 # 注意：如果 Dataset 返回的是 [B, T, C, H, W] (叠在一起了)，需要先 reshape
#                 # 这里假设您的 FusionEncoder 接口处理好了多视角拼接
#                 # 下面是一个通用的 Mask 逻辑：
                
#                 # 如果是双摄拼接在 Channel 维 (C=6)，前3通道是 Main
#                 if video.shape[-3] == 6:
#                     video[:, :, :3, :, :] = 0.0
                
#                 # 如果是单独的维度 (B, 2, T, C, H, W)
#                 elif video.ndim == 6 and video.shape[1] == 2:
#                     video[:, 0] = 0.0 # Mask Main View


#             optimizer.zero_grad()

#             with autocast('cuda', dtype=torch.bfloat16):
#                 with torch.no_grad():
#                     encoder_outputs = fusion_encoder(video, text, state, ff)
#                     e_t = encoder_outputs['e_t'] 
                
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 conditions = {"e_t": e_t}
#                 pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
#                 if pred_noise.dim() == 3 and noise.dim() == 2:
#                     noise = noise.unsqueeze(1) 
                    
#                 loss = torch.nn.functional.mse_loss(pred_noise, noise)

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             if i % 10 == 0 and i > 0:
#                 elapsed = time.time() - start_time
#                 speed = (i * args.batch_size) / elapsed
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f} | Speed: {speed:.1f} samples/s")

#         # 保存
#         os.makedirs(args.output_dir, exist_ok=True)
#         # 注意：使用 LoRA 时，state_dict 会只包含 LoRA 权重，文件很小
#         if epoch % 10 == 0 or epoch == args.epochs - 1:
#             save_path = os.path.join(args.output_dir, f"stageC_lora_epoch_{epoch}.pt")
#             torch.save({
#                 'rdt_state_dict': rdt_wrapper.state_dict(), # 包含 LoRA 权重
#                 'encoder_state_dict': fusion_encoder.state_dict()
#             }, save_path)
#             print(f"Saved Checkpoint to {save_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
#     parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/projects/checkpoints/stageB_papercup.pt')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/projects/checkpoints')
#     parser.add_argument('--batch_size', type=int, default=48)
#     parser.add_argument('--num_workers', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=50) # LoRA 建议多训几轮
    
#     args = parser.parse_args()
#     train_stage_c(args)

# train/stageC_joint.py
import sys
import os
import torch
import torch.optim as optim
import argparse
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
# === [新增] 强制开启 Flash Attention (A100 必备) ===
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False) # 禁止普通数学 Attention，防止 OOM
print(f"Flash Attention Enabled: {torch.backends.cuda.flash_sdp_enabled()}")
# 添加项目根目录到路径，确保能导入 models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_encoder import FusionEncoder
from models.rdt_model import RDTWrapper
from utils.dataset_loader import RobotDataset
from losses.consistency_loss import compute_consistency_loss

# === 路径配置 (请根据实际情况调整) ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = '/yanghaochuan/data/1223dataset_stats.json'

def train_stage_c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage C Joint Training (End-to-End) ===")
    
    # ====================================================
    # 1. 初始化 Fusion Encoder
    # ====================================================
    print("Loading Fusion Encoder...")
    # rdt_dim=768 必须与 rdt_model.py 中 visual_proj 的输入维度一致
    fusion_encoder = FusionEncoder(
        backbone_path=VIDEO_MAE_PATH, 
        teacher_dim=1152,
        rdt_dim=768 
    ).to(device)
    
    # 加载 Stage B 预训练权重 (如果有)
    if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
        print(f"Loading Stage B Checkpoint: {args.stage_b_ckpt}")
        try:
            ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
            # 兼容保存时可能只保存了 encoder_state_dict 或 整个 dict 的情况
            state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
            fusion_encoder.load_state_dict(state_dict, strict=False)
            print("✅ Stage B weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load Stage B weights: {e}")
    
    # 冻结 VideoMAE Backbone，只训练 Adapter 和 Heads
    fusion_encoder.eval() 
    for param in fusion_encoder.parameters(): param.requires_grad = True 
    for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
    if fusion_encoder.text_encoder:
        for param in fusion_encoder.text_encoder.parameters(): param.requires_grad = False

    # ====================================================
    # 2. 初始化 RDT Policy (Wrapper)
    # ====================================================
    print("Loading RDT Policy...")
    rdt_wrapper = RDTWrapper(
        action_dim=8, 
        model_path=RDT_PATH, 
        pred_horizon=args.pred_horizon
    ).to(device)
    
    # ====================================================
    # 3. 配置 LoRA
    # ====================================================
    print("Applying LoRA to RDT...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
        lora_dropout=0.05, 
        bias="none"
    )
    # 将 LoRA 挂载到 RDT 内部的 Transformer 上
    rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    rdt_wrapper.rdt_model.print_trainable_parameters()
    
    # ====================================================
    # 4. 优化器 & 调度器
    # ====================================================
    # 联合训练：同时优化 RDT 的 LoRA 参数 和 FusionEncoder 的 Adapter 参数
    params_to_optimize = [
        {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
    ]
    optimizer = optim.AdamW(params_to_optimize, weight_decay=1e-4)
    
    # 噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        beta_schedule="squaredcos_cap_v2", 
        prediction_type="sample"
    )

    # ====================================================
    # 5. 数据加载
    # ====================================================
    print(f"Loading Dataset from {args.data_root}")
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data file not found: {args.data_root}")

    dataset = RobotDataset(
        hdf5_path=args.data_root, 
        window_size=16, 
        pred_horizon=args.pred_horizon, 
        stats_path=STATS_PATH
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=True
    )

    # ====================================================
    # 6. 训练循环
    # ====================================================
    rdt_wrapper.train()
    fusion_encoder.train()
    
    print(">>> Training Started... <<<")
    
    for epoch in range(args.epochs):
        total_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            # 获取数据
            # video shape: [B, 2, 3, 16, H, W] (View 0=Main, View 1=Wrist)
            video = batch['video'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True) # [B, 16, 8]
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            actions = batch['action_target'].to(device, non_blocking=True)

            # === Modality Dropout ===
            # # 50% 概率将 Main Camera (View 0) 全黑，强迫模型学会看 Wrist 和 听指令
            # if torch.rand(1) < 0.5:
            #      video[:, 0] = 0.0 
            rand_val = torch.rand(1).item()
            if rand_val < 0.7:
                 video[:, 0] = 0.0  # Mask Main View
            
            # 策略 B: 10% 的时间把手腕抹黑 (防止过拟合，可选)
            elif rand_val < 0.8:
                 video[:, 1] = 0.0
            
            optimizer.zero_grad()
            
            with autocast('cuda', dtype=torch.bfloat16):
                # -------------------------
                # A. 提取视觉特征
                # -------------------------
                encoder_outputs = fusion_encoder(video, text, state, ff)
                e_t = encoder_outputs['e_t'] # [B, 64, 768]
                
                # -------------------------
                # B. 计算 Diffusion Loss
                # -------------------------
                # 随机采样时间步
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, 
                    (actions.shape[0],), device=device
                ).long()
                
                # 加噪
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                # [核心修改] 构造 Conditions 字典
                # 必须传入 'state'，RDTWrapper 会将其作为 State Token
                # 我们取历史窗口的最后一帧 (current state) 作为条件
                current_state = state[:, -1, :] # [B, 8]
                
                conditions = {
                    "e_t": e_t, 
                    "state": current_state 
                }
                
                # 预测噪声
                pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                loss_diff = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # -------------------------
                # C. 计算 Consistency Loss
                # -------------------------
                # 确保特征在单摄/双摄情况下保持一致
                loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                
                # 总 Loss
                total_loss_step = loss_diff + 0.1 * loss_cons

            # 反向传播
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
            optimizer.step()
            
            total_loss += total_loss_step.item()
            
            # 打印日志
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} [{i}/{len(loader)}] Diff: {loss_diff.item():.4f} Cons: {loss_cons.item():.4f} Total: {total_loss_step.item():.4f}")

        # 保存 Checkpoint
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"1223stageC_joint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'rdt_state_dict': rdt_wrapper.state_dict(),     # 包含 LoRA 权重
                'encoder_state_dict': fusion_encoder.state_dict(), # 包含 Adapter 权重
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"✅ Saved checkpoint to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--stage_b_ckpt', type=str, default=None, help="Path to Stage B pretrained checkpoint")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--pred_horizon', type=int, default=16, help="Prediction horizon for action chunking")
    
    args = parser.parse_args()
    
    # 简单的参数检查
    if not args.stage_b_ckpt:
        print("⚠️ Warning: No Stage B checkpoint provided. FusionEncoder will be initialized randomly (except Backbone).")
        
    train_stage_c(args)