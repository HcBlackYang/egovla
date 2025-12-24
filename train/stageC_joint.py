# import sys
# import os
# import torch
# import torch.optim as optim
# import argparse
# import time
# import numpy as np
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.amp import autocast
# from diffusers import DDPMScheduler
# from peft import LoraConfig, get_peft_model
# from torch.utils.tensorboard import SummaryWriter # === [新增] TensorBoard

# # === [新增] WandB 检查与导入 ===
# try:
#     import wandb
#     HAS_WANDB = True
# except ImportError:
#     HAS_WANDB = False
#     print("⚠️ WandB not found. Install with `pip install wandb` for better visualization.")

# # === 性能优化 ===
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.enable_math_sdp(False)

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.fusion_encoder import FusionEncoder
# from model.rdt_model import RDTWrapper
# from utils.dataset_loader import RobotDataset
# from losses.consistency_loss import compute_consistency_loss

# # === 路径配置 ===
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = '/yanghaochuan/data/1223dataset_stats.json'

# def train_stage_c(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # ====================================================
#     # 0. 初始化可视化记录器 (TensorBoard + WandB)
#     # ====================================================
#     # TensorBoard 日志目录 (通常服务器网页会读取这个路径)
#     log_dir = os.path.join(args.output_dir, "logs")
#     os.makedirs(log_dir, exist_ok=True)
#     tb_writer = SummaryWriter(log_dir=log_dir)
#     print(f"📈 TensorBoard logging to: {log_dir}")
    
#     # WandB 初始化
#     if args.use_wandb and HAS_WANDB:
#         wandb.init(
#             project="RDT-StageC-Joint",
#             name=f"run_horizon{args.pred_horizon}_{int(time.time())}",
#             config=vars(args)
#         )
#         print("🚀 WandB logging enabled.")
    
#     print(f"=== Stage C Joint Training ===")
    
#     # ====================================================
#     # 1. 模型加载 (FusionEncoder + RDT)
#     # ====================================================
#     print("Loading Models...")
#     fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
#     # 加载 Stage B
#     if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B: {args.stage_b_ckpt}")
#         ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
#         state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
#         fusion_encoder.load_state_dict(state_dict, strict=False)
    
#     # 冻结 VideoMAE
#     fusion_encoder.eval() 
#     for param in fusion_encoder.parameters(): param.requires_grad = True 
#     for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
#     if fusion_encoder.text_encoder:
#         for p in fusion_encoder.text_encoder.parameters(): p.requires_grad = False

#     # RDT Wrapper
#     rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, pred_horizon=args.pred_horizon).to(device)
    
#     # RDT 权重自动切片加载
#     if os.path.exists(RDT_PATH) or os.path.exists(os.path.join(RDT_PATH, "pytorch_model.bin")):
#         rdt_file = RDT_PATH if os.path.isfile(RDT_PATH) else os.path.join(RDT_PATH, "pytorch_model.bin")
#         if os.path.exists(rdt_file):
#             print("Loading RDT weights with auto-slicing...")
#             state_dict = torch.load(rdt_file, map_location='cpu')
#             if 'x_pos_embed' in state_dict:
#                 ckpt_pos = state_dict['x_pos_embed']
#                 curr_pos = rdt_wrapper.rdt_model.x_pos_embed
#                 if ckpt_pos.shape != curr_pos.shape:
#                     print(f"✂️ Slicing position embed: {ckpt_pos.shape} -> {curr_pos.shape}")
#                     state_dict['x_pos_embed'] = ckpt_pos[:, :curr_pos.shape[1], :]
#             rdt_wrapper.rdt_model.load_state_dict(state_dict, strict=False)

#     # LoRA
#     print("Applying LoRA...")
#     peft_config = LoraConfig(
#         r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
#         lora_dropout=0.05, bias="none"
#     )
#     rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    
#     # ====================================================
#     # 2. 优化器 & 数据
#     # ====================================================
#     params = [
#         {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
#         {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
#     ]
#     optimizer = optim.AdamW(params, weight_decay=1e-4)
#     noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

#     print(f"Loading Dataset from {args.data_root}")
#     dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, pred_horizon=args.pred_horizon, stats_path=STATS_PATH)
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

#     # ====================================================
#     # 3. 训练循环 (带可视化)
#     # ====================================================
#     print(">>> Training Started <<<")
#     global_step = 0
    
#     for epoch in range(args.epochs):
#         rdt_wrapper.train()
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             video = batch['video'].to(device, non_blocking=True) # [B, 2, 3, 16, H, W]
#             state = batch['state'].to(device, non_blocking=True)
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
#             actions = batch['action_target'].to(device, non_blocking=True)

#             # Modality Dropout
#             rand_val = torch.rand(1).item()
#             mask_type = "None"
#             if rand_val < 0.7:
#                  video[:, 0] = 0.0  # Mask Main
#                  mask_type = "Main_Masked"
#             elif rand_val < 0.8:
#                  video[:, 1] = 0.0  # Mask Wrist
#                  mask_type = "Wrist_Masked"
            
#             optimizer.zero_grad()
            
#             with autocast('cuda', dtype=torch.bfloat16):
#                 # Forward
#                 e_t = fusion_encoder(video, text, state, ff)['e_t']
                
#                 # Loss
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 conditions = {"e_t": e_t, "state": state[:, -1, :]}
#                 pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
#                 loss_diff = F.mse_loss(pred_noise, noise)
#                 loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
#                 total_loss = loss_diff + 0.1 * loss_cons

#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
#             optimizer.step()
            
#             # --- 日志记录 ---
#             if i % 10 == 0:
#                 # 1. 打印到控制台
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {total_loss.item():.4f} (Diff: {loss_diff.item():.4f} Cons: {loss_cons.item():.4f})")
                
#                 # 2. 写入 TensorBoard
#                 tb_writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
#                 tb_writer.add_scalar('Train/Diff_Loss', loss_diff.item(), global_step)
#                 tb_writer.add_scalar('Train/Cons_Loss', loss_cons.item(), global_step)
                
#                 # 3. 写入 WandB
#                 if args.use_wandb and HAS_WANDB:
#                     wandb.log({
#                         "total_loss": total_loss.item(),
#                         "diff_loss": loss_diff.item(),
#                         "cons_loss": loss_cons.item(),
#                         "epoch": epoch
#                     }, step=global_step)
            
#             # --- 视频可视化 (每 500 步一次) ---
#             if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
#                 # 提取第一个样本的视频: [2, 3, 16, H, W]
#                 # View 0: Main, View 1: Wrist
#                 vid_sample = video[0].float().cpu().numpy() # [2, 3, 16, H, W]
                
#                 # 转换为 GIF 格式 [T, C, H, W] -> wandb 需 [T, C, H, W]
#                 # 我们把 Main 和 Wrist 拼在一起显示
#                 main_view = vid_sample[0] # [3, 16, H, W] -> [16, 3, H, W]
#                 wrist_view = vid_sample[1]
                
#                 # 处理一下维度顺序给 wandb: (Time, Channel, Height, Width)
#                 main_view = np.transpose(main_view, (1, 0, 2, 3))
#                 wrist_view = np.transpose(wrist_view, (1, 0, 2, 3))
                
#                 # 拼接: 左右拼接
#                 combined_view = np.concatenate([main_view, wrist_view], axis=3) # Width 维度拼接
                
#                 # 记录视频
#                 wandb.log({
#                     "input_video": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"E{epoch}-S{i}: {mask_type}")
#                 }, step=global_step)
#                 print("🎥 Video sample uploaded to WandB.")

#             global_step += 1

#         # 保存 Checkpoint
#         if epoch % 5 == 0 or epoch == args.epochs - 1:
#             save_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
#             torch.save({
#                 'epoch': epoch,
#                 'rdt_state_dict': rdt_wrapper.state_dict(),
#                 'encoder_state_dict': fusion_encoder.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'pred_horizon': args.pred_horizon
#             }, save_path)
#             print(f"✅ Saved to {save_path}")

#     tb_writer.close()
#     if args.use_wandb and HAS_WANDB:
#         wandb.finish()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
#     parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/checkpoints/1223stageB_papercup.pt')
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--pred_horizon', type=int, default=64)
    
#     # === [新增] 可视化开关 ===
#     parser.add_argument('--use_wandb', action='store_true', default=False, help="Enable WandB logging")
    
#     args = parser.parse_args()
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
from torch.utils.tensorboard import SummaryWriter 

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
    # TensorBoard 日志目录
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print(f"📈 TensorBoard logging to: {log_dir}")
    
    # WandB 初始化
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="RDT-StageC-Joint",
            name=f"run_horizon{args.pred_horizon}_{int(time.time())}",
            config=vars(args),
            resume="allow" # 允许 WandB 断点续传
        )
        print("🚀 WandB logging enabled.")
    
    print(f"=== Stage C Joint Training ===")
    
    # ====================================================
    # 1. 模型加载 (FusionEncoder + RDT)
    # ====================================================
    print("Loading Models...")
    fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
    # 加载 Stage B (如果有)
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
    
    # RDT 权重自动切片加载 (处理 Horizon 16 vs 64 的问题)
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

    # LoRA 配置
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
        lora_dropout=0.05, bias="none"
    )
    rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    
    # ====================================================
    # 2. 优化器 & 调度器
    # ====================================================
    params = [
        {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

    # ====================================================
    # 🌟 [关键修改] 断点续训逻辑 (Resume Training)
    # ====================================================
    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"🔄 Resuming training from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            
            # 1. 恢复 Epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"   -> Resuming from Epoch {start_epoch}")
            
            # 2. 恢复 RDT Policy (包含 LoRA 权重)
            if 'rdt_state_dict' in checkpoint:
                rdt_wrapper.load_state_dict(checkpoint['rdt_state_dict'], strict=False)
                print("   -> RDT Policy (LoRA) weights restored.")
            
            # 3. 恢复 Fusion Encoder (微调后的权重)
            if 'encoder_state_dict' in checkpoint:
                fusion_encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
                print("   -> Fusion Encoder weights restored.")
                
            # 4. 恢复 优化器状态 (保证动量不丢失)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("   -> Optimizer state restored.")
                except Exception as e:
                    print(f"   ⚠️ Optimizer load failed (skipping): {e}")
        else:
            print(f"⚠️ Checkpoint file not found: {args.resume_from_checkpoint}")

    # ====================================================
    # 3. 数据加载
    # ====================================================
    print(f"Loading Dataset from {args.data_root}")
    dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, pred_horizon=args.pred_horizon, stats_path=STATS_PATH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # ====================================================
    # 4. 训练循环 (带可视化)
    # ====================================================
    print(">>> Training Started <<<")
    # 修正 global_step 以匹配 resume
    global_step = start_epoch * len(loader)
    
    # 循环从 start_epoch 开始
    for epoch in range(start_epoch, args.epochs):
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
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {total_loss.item():.4f} (Diff: {loss_diff.item():.4f} Cons: {loss_cons.item():.4f})")
                
                tb_writer.add_scalar('Train/Total_Loss', total_loss.item(), global_step)
                tb_writer.add_scalar('Train/Diff_Loss', loss_diff.item(), global_step)
                tb_writer.add_scalar('Train/Cons_Loss', loss_cons.item(), global_step)
                
                if args.use_wandb and HAS_WANDB:
                    wandb.log({
                        "total_loss": total_loss.item(),
                        "diff_loss": loss_diff.item(),
                        "cons_loss": loss_cons.item(),
                        "epoch": epoch
                    }, step=global_step)
            
            # --- 视频可视化 (每 500 步一次) ---
            if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
                try:
                    vid_sample = video[0].float().cpu().numpy() 
                    main_view = np.transpose(vid_sample[0], (1, 0, 2, 3)) # [T, C, H, W]
                    wrist_view = np.transpose(vid_sample[1], (1, 0, 2, 3))
                    combined_view = np.concatenate([main_view, wrist_view], axis=3) 
                    
                    wandb.log({
                        "input_video": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"E{epoch}-S{i}: {mask_type}")
                    }, step=global_step)
                    print("🎥 Video sample uploaded to WandB.")
                except Exception as e:
                    print(f"⚠️ Video log failed: {e}")

            global_step += 1

            # ====================================================
            # 🛑 [新增] 达到最大步数强制停止 (Max Train Steps)
            # ====================================================
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                print(f"🛑 Reached max_train_steps ({args.max_train_steps}). Stopping training.")
                
                # 强制保存当前 Checkpoint
                save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'rdt_state_dict': rdt_wrapper.state_dict(),
                    'encoder_state_dict': fusion_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'pred_horizon': args.pred_horizon
                }, save_path)
                print(f"✅ Final checkpoint saved to {save_path}")
                
                # 关闭记录器并退出
                tb_writer.close()
                if args.use_wandb and HAS_WANDB:
                    wandb.finish()
                return # 退出训练函数

        # 保存 Checkpoint (每 2 个 Epoch 或 最后一个)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
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
    
    # === 可视化 & 续训 & 控制 ===
    parser.add_argument('--use_wandb', action='store_true', default=False, help="Enable WandB logging")
    # 续训参数
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help="Path to checkpoint to resume from (e.g. checkpoints/epoch_10.pt)")
    # [新增] 最大步数停止参数
    parser.add_argument('--max_train_steps', type=int, default=None, 
                        help="Force stop training after this many global steps (e.g. 2500).")
    
    args = parser.parse_args()
    train_stage_c(args)