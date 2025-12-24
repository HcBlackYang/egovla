# # train/stageC_joint.py
# import sys
# import os
# import torch
# import torch.optim as optim
# import argparse
# import time
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.amp import autocast
# from diffusers import DDPMScheduler
# from peft import LoraConfig, get_peft_model
# # === [新增] 强制开启 Flash Attention (A100 必备) ===
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_math_sdp(False) # 禁止普通数学 Attention，防止 OOM
# print(f"Flash Attention Enabled: {torch.backends.cuda.flash_sdp_enabled()}")
# # 添加项目根目录到路径，确保能导入 models
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper
# from utils.dataset_loader import RobotDataset
# from losses.consistency_loss import compute_consistency_loss

# # === 路径配置 (请根据实际情况调整) ===
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = '/yanghaochuan/data/1223dataset_stats.json'

# def train_stage_c(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Stage C Joint Training (End-to-End) ===")
    
#     # ====================================================
#     # 1. 初始化 Fusion Encoder
#     # ====================================================
#     print("Loading Fusion Encoder...")
#     # rdt_dim=768 必须与 rdt_model.py 中 visual_proj 的输入维度一致
#     fusion_encoder = FusionEncoder(
#         backbone_path=VIDEO_MAE_PATH, 
#         teacher_dim=1152,
#         rdt_dim=768 
#     ).to(device)
    
#     # 加载 Stage B 预训练权重 (如果有)
#     if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B Checkpoint: {args.stage_b_ckpt}")
#         try:
#             ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
#             # 兼容保存时可能只保存了 encoder_state_dict 或 整个 dict 的情况
#             state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
#             fusion_encoder.load_state_dict(state_dict, strict=False)
#             print("✅ Stage B weights loaded successfully.")
#         except Exception as e:
#             print(f"⚠️ Failed to load Stage B weights: {e}")
    
#     # 冻结 VideoMAE Backbone，只训练 Adapter 和 Heads
#     fusion_encoder.eval() 
#     for param in fusion_encoder.parameters(): param.requires_grad = True 
#     for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
#     if fusion_encoder.text_encoder:
#         for param in fusion_encoder.text_encoder.parameters(): param.requires_grad = False

#     # ====================================================
#     # 2. 初始化 RDT Policy (Wrapper)
#     # ====================================================
#     print("Loading RDT Policy...")
#     rdt_wrapper = RDTWrapper(
#         action_dim=8, 
#         model_path=RDT_PATH, 
#         pred_horizon=args.pred_horizon
#     ).to(device)
    
#     # ====================================================
#     # 3. 配置 LoRA
#     # ====================================================
#     print("Applying LoRA to RDT...")
#     peft_config = LoraConfig(
#         r=16, 
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
#         lora_dropout=0.05, 
#         bias="none"
#     )
#     # 将 LoRA 挂载到 RDT 内部的 Transformer 上
#     rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
#     rdt_wrapper.rdt_model.print_trainable_parameters()
    
#     # ====================================================
#     # 4. 优化器 & 调度器
#     # ====================================================
#     # 联合训练：同时优化 RDT 的 LoRA 参数 和 FusionEncoder 的 Adapter 参数
#     params_to_optimize = [
#         {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
#         {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
#     ]
#     optimizer = optim.AdamW(params_to_optimize, weight_decay=1e-4)
    
#     # 噪声调度器
#     noise_scheduler = DDPMScheduler(
#         num_train_timesteps=1000, 
#         beta_schedule="squaredcos_cap_v2", 
#         prediction_type="sample"
#     )

#     # ====================================================
#     # 5. 数据加载
#     # ====================================================
#     print(f"Loading Dataset from {args.data_root}")
#     if not os.path.exists(args.data_root):
#         raise FileNotFoundError(f"Data file not found: {args.data_root}")

#     dataset = RobotDataset(
#         hdf5_path=args.data_root, 
#         window_size=16, 
#         pred_horizon=args.pred_horizon, 
#         stats_path=STATS_PATH
#     )
    
#     loader = DataLoader(
#         dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         num_workers=8, 
#         pin_memory=True, 
#         drop_last=True
#     )

#     # ====================================================
#     # 6. 训练循环
#     # ====================================================
#     rdt_wrapper.train()
#     fusion_encoder.train()
    
#     print(">>> Training Started... <<<")
    
#     for epoch in range(args.epochs):
#         total_loss = 0
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             # 获取数据
#             # video shape: [B, 2, 3, 16, H, W] (View 0=Main, View 1=Wrist)
#             video = batch['video'].to(device, non_blocking=True)
#             state = batch['state'].to(device, non_blocking=True) # [B, 16, 8]
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
#             actions = batch['action_target'].to(device, non_blocking=True)

#             # === Modality Dropout ===
#             # # 50% 概率将 Main Camera (View 0) 全黑，强迫模型学会看 Wrist 和 听指令
#             # if torch.rand(1) < 0.5:
#             #      video[:, 0] = 0.0 
#             rand_val = torch.rand(1).item()
#             if rand_val < 0.7:
#                  video[:, 0] = 0.0  # Mask Main View
            
#             # 策略 B: 10% 的时间把手腕抹黑 (防止过拟合，可选)
#             elif rand_val < 0.8:
#                  video[:, 1] = 0.0
            
#             optimizer.zero_grad()
            
#             with autocast('cuda', dtype=torch.bfloat16):
#                 # -------------------------
#                 # A. 提取视觉特征
#                 # -------------------------
#                 encoder_outputs = fusion_encoder(video, text, state, ff)
#                 e_t = encoder_outputs['e_t'] # [B, 64, 768]
                
#                 # -------------------------
#                 # B. 计算 Diffusion Loss
#                 # -------------------------
#                 # 随机采样时间步
#                 timesteps = torch.randint(
#                     0, noise_scheduler.config.num_train_timesteps, 
#                     (actions.shape[0],), device=device
#                 ).long()
                
#                 # 加噪
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 # [核心修改] 构造 Conditions 字典
#                 # 必须传入 'state'，RDTWrapper 会将其作为 State Token
#                 # 我们取历史窗口的最后一帧 (current state) 作为条件
#                 current_state = state[:, -1, :] # [B, 8]
                
#                 conditions = {
#                     "e_t": e_t, 
#                     "state": current_state 
#                 }
                
#                 # 预测噪声
#                 pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
#                 loss_diff = torch.nn.functional.mse_loss(pred_noise, noise)
                
#                 # -------------------------
#                 # C. 计算 Consistency Loss
#                 # -------------------------
#                 # 确保特征在单摄/双摄情况下保持一致
#                 loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                
#                 # 总 Loss
#                 total_loss_step = loss_diff + 0.1 * loss_cons

#             # 反向传播
#             total_loss_step.backward()
#             torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
#             optimizer.step()
            
#             total_loss += total_loss_step.item()
            
#             # 打印日志
#             if i % 10 == 0:
#                 elapsed = time.time() - start_time
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] Diff: {loss_diff.item():.4f} Cons: {loss_cons.item():.4f} Total: {total_loss_step.item():.4f}")

#         # 保存 Checkpoint
#         if epoch % 5 == 0 or epoch == args.epochs - 1:
#             os.makedirs(args.output_dir, exist_ok=True)
#             save_path = os.path.join(args.output_dir, f"1223stageC_joint_epoch_{epoch}.pt")
#             torch.save({
#                 'epoch': epoch,
#                 'rdt_state_dict': rdt_wrapper.state_dict(),     # 包含 LoRA 权重
#                 'encoder_state_dict': fusion_encoder.state_dict(), # 包含 Adapter 权重
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }, save_path)
#             print(f"✅ Saved checkpoint to {save_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, required=True, help="Path to HDF5 dataset")
#     parser.add_argument('--output_dir', type=str, default='./checkpoints', help="Directory to save checkpoints")
#     parser.add_argument('--stage_b_ckpt', type=str, default=None, help="Path to Stage B pretrained checkpoint")
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--pred_horizon', type=int, default=16, help="Prediction horizon for action chunking")
    
#     args = parser.parse_args()
    
#     # 简单的参数检查
#     if not args.stage_b_ckpt:
#         print("⚠️ Warning: No Stage B checkpoint provided. FusionEncoder will be initialized randomly (except Backbone).")
        
#     train_stage_c(args)

import sys
import os
import torch
import torch.optim as optim
import argparse
import time
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter # === [新增] TensorBoard

# === [新增] WandB 检查与导入 ===
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️ WandB not found. Install with `pip install wandb` for better visualization.")

# === 性能优化 ===
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fusion_encoder import FusionEncoder
from model.rdt_model import RDTWrapper
from utils.dataset_loader import RobotDataset
from losses.consistency_loss import compute_consistency_loss

# === 路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = '/yanghaochuan/data/1223dataset_stats.json'

def train_stage_c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ====================================================
    # 0. 初始化可视化记录器 (TensorBoard + WandB)
    # ====================================================
    # TensorBoard 日志目录 (通常服务器网页会读取这个路径)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print(f"📈 TensorBoard logging to: {log_dir}")
    
    # WandB 初始化
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="RDT-StageC-Joint",
            name=f"run_horizon{args.pred_horizon}_{int(time.time())}",
            config=vars(args)
        )
        print("🚀 WandB logging enabled.")
    
    print(f"=== Stage C Joint Training ===")
    
    # ====================================================
    # 1. 模型加载 (FusionEncoder + RDT)
    # ====================================================
    print("Loading Models...")
    fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
    # 加载 Stage B
    if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
        print(f"Loading Stage B: {args.stage_b_ckpt}")
        ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
        state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
        fusion_encoder.load_state_dict(state_dict, strict=False)
    
    # 冻结 VideoMAE
    fusion_encoder.eval() 
    for param in fusion_encoder.parameters(): param.requires_grad = True 
    for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
    if fusion_encoder.text_encoder:
        for p in fusion_encoder.text_encoder.parameters(): p.requires_grad = False

    # RDT Wrapper
    rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, pred_horizon=args.pred_horizon).to(device)
    
    # RDT 权重自动切片加载
    if os.path.exists(RDT_PATH) or os.path.exists(os.path.join(RDT_PATH, "pytorch_model.bin")):
        rdt_file = RDT_PATH if os.path.isfile(RDT_PATH) else os.path.join(RDT_PATH, "pytorch_model.bin")
        if os.path.exists(rdt_file):
            print("Loading RDT weights with auto-slicing...")
            state_dict = torch.load(rdt_file, map_location='cpu')
            if 'x_pos_embed' in state_dict:
                ckpt_pos = state_dict['x_pos_embed']
                curr_pos = rdt_wrapper.rdt_model.x_pos_embed
                if ckpt_pos.shape != curr_pos.shape:
                    print(f"✂️ Slicing position embed: {ckpt_pos.shape} -> {curr_pos.shape}")
                    state_dict['x_pos_embed'] = ckpt_pos[:, :curr_pos.shape[1], :]
            rdt_wrapper.rdt_model.load_state_dict(state_dict, strict=False)

    # LoRA
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
        lora_dropout=0.05, bias="none"
    )
    rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    
    # ====================================================
    # 2. 优化器 & 数据
    # ====================================================
    params = [
        {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

    print(f"Loading Dataset from {args.data_root}")
    dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, pred_horizon=args.pred_horizon, stats_path=STATS_PATH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # ====================================================
    # 3. 训练循环 (带可视化)
    # ====================================================
    print(">>> Training Started <<<")
    global_step = 0
    
    for epoch in range(args.epochs):
        rdt_wrapper.train()
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            video = batch['video'].to(device, non_blocking=True) # [B, 2, 3, 16, H, W]
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            actions = batch['action_target'].to(device, non_blocking=True)

            # Modality Dropout
            rand_val = torch.rand(1).item()
            mask_type = "None"
            if rand_val < 0.7:
                 video[:, 0] = 0.0  # Mask Main
                 mask_type = "Main_Masked"
            elif rand_val < 0.8:
                 video[:, 1] = 0.0  # Mask Wrist
                 mask_type = "Wrist_Masked"
            
            optimizer.zero_grad()
            
            with autocast('cuda', dtype=torch.bfloat16):
                # Forward
                e_t = fusion_encoder(video, text, state, ff)['e_t']
                
                # Loss
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                conditions = {"e_t": e_t, "state": state[:, -1, :]}
                pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
                loss_diff = F.mse_loss(pred_noise, noise)
                loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                total_loss = loss_diff + 0.1 * loss_cons

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
            optimizer.step()
            
            # --- 日志记录 ---
            if i % 10 == 0:
                # 1. 打印到控制台
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {total_loss.item():.4f} (Diff: {loss_diff.item():.4f} Cons: {loss_cons.item():.4f})")
                
                # 2. 写入 TensorBoard
                tb_writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                tb_writer.add_scalar('Train/Diff_Loss', loss_diff.item(), global_step)
                tb_writer.add_scalar('Train/Cons_Loss', loss_cons.item(), global_step)
                
                # 3. 写入 WandB
                if args.use_wandb and HAS_WANDB:
                    wandb.log({
                        "total_loss": total_loss.item(),
                        "diff_loss": loss_diff.item(),
                        "cons_loss": loss_cons.item(),
                        "epoch": epoch
                    }, step=global_step)
            
            # --- 视频可视化 (每 500 步一次) ---
            if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
                # 提取第一个样本的视频: [2, 3, 16, H, W]
                # View 0: Main, View 1: Wrist
                vid_sample = video[0].float().cpu().numpy() # [2, 3, 16, H, W]
                
                # 转换为 GIF 格式 [T, C, H, W] -> wandb 需 [T, C, H, W]
                # 我们把 Main 和 Wrist 拼在一起显示
                main_view = vid_sample[0] # [3, 16, H, W] -> [16, 3, H, W]
                wrist_view = vid_sample[1]
                
                # 处理一下维度顺序给 wandb: (Time, Channel, Height, Width)
                main_view = np.transpose(main_view, (1, 0, 2, 3))
                wrist_view = np.transpose(wrist_view, (1, 0, 2, 3))
                
                # 拼接: 左右拼接
                combined_view = np.concatenate([main_view, wrist_view], axis=3) # Width 维度拼接
                
                # 记录视频
                wandb.log({
                    "input_video": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"E{epoch}-S{i}: {mask_type}")
                }, step=global_step)
                print("🎥 Video sample uploaded to WandB.")

            global_step += 1

        # 保存 Checkpoint
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'rdt_state_dict': rdt_wrapper.state_dict(),
                'encoder_state_dict': fusion_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pred_horizon': args.pred_horizon
            }, save_path)
            print(f"✅ Saved to {save_path}")

    tb_writer.close()
    if args.use_wandb and HAS_WANDB:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
    parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/checkpoints/1223stageB_papercup.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--pred_horizon', type=int, default=64)
    
    # === [新增] 可视化开关 ===
    parser.add_argument('--use_wandb', action='store_true', default=True, help="Enable WandB logging")
    
    args = parser.parse_args()
    train_stage_c(args)