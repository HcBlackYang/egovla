# # train/stageB_train.py
# import sys
# import os
# import torch
# import torch.optim as optim
# import argparse
# import time
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.amp import autocast # BF16 不需要 GradScaler

# # === [环境检查] WandB ===
# try:
#     import wandb
#     HAS_WANDB = True
# except ImportError:
#     HAS_WANDB = False
#     print("⚠️ WandB not found. Install with `pip install wandb`")

# # 强制使用 Flash Attention
# torch.backends.cuda.enable_flash_sdp(True)
# torch.backends.cuda.enable_math_sdp(False) 
# torch.backends.cuda.enable_mem_efficient_sdp(True)

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.fusion_encoder import FusionEncoder
# from losses.distillation_loss import DistillationLoss
# from losses.decoupling_regularizer import DecouplingLoss
# from losses.temporal_consistency import TemporalConsistencyLoss
# from utils.dataset_loader import RobotDataset

# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'

# def train_stage_b(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Stage B Training: Real Teacher Distillation on {device} ===")
#     print(f"=== Mode: Step-Based Training | Target: {args.max_train_steps} Steps ===")
    
#     # 启用 TF32
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     # === WandB 初始化 ===
#     if args.use_wandb and HAS_WANDB:
#         wandb.init(
#             project="RDT-StageB-Pretrain",
#             name=f"step{args.max_train_steps}_mask0.8_{int(time.time())}",
#             config=vars(args),
#             resume="allow"
#         )

#     # 1. 初始化模型
#     model = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
#     if os.path.exists(args.stage_a_ckpt):
#         print(f"Loading Stage A: {args.stage_a_ckpt}")
#         model.load_state_dict(torch.load(args.stage_a_ckpt), strict=False)

#     # 冻结 VideoMAE，只微调 Adapter 和部分 Block
#     for param in model.backbone.parameters(): param.requires_grad = False
    
#     layers_to_train = ["blocks.20", "blocks.21", "blocks.22", "blocks.23"] 
#     count = 0
#     for name, param in model.backbone.named_parameters():
#         if any(x in name for x in layers_to_train) or "encoder.layer.2" in name: 
#             param.requires_grad = True
#             count += 1
#     print(f"Unfrozen {count} parameters in VideoMAE backbone.")
    
#     for p in model.routing_layer.parameters(): p.requires_grad = True
#     for p in model.semantic_align_head.parameters(): p.requires_grad = True
#     for p in model.temporal_align_head.parameters(): p.requires_grad = True
#     for p in model.projection_head.parameters(): p.requires_grad = True
    
#     print("Compiling model with torch.compile...")
#     # try:
#     #     model = torch.compile(model)
#     # except Exception as e:
#     #     print(f"Compile failed: {e}")

#     # 2. 数据加载
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

#     # 3. 损失与优化
#     distill_fn = DistillationLoss()
#     decouple_fn = DecouplingLoss()
#     temporal_fn = TemporalConsistencyLoss()
    
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
#     # loss_weights = {"distill":1.0, "decouple":0.5, "consistency":1.0}
#     loss_weights = {"distill":1.0, "decouple":0.5, "consistency":0.1}

#     # ====================================================
#     # 4. 精准断点续训逻辑 (Step-Based Resume)
#     # ====================================================
#     global_step = 0
#     start_epoch = 0
#     resume_batch_idx = 0

#     if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
#         print(f"🔄 Resuming from: {args.resume_from_checkpoint}")
#         checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
#         # 恢复权重
#         # 兼容只保存了 model state dict 的情况，也兼容保存了完整 info 的情况
#         if 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         else:
#             model.load_state_dict(checkpoint) # 旧格式

#         if 'optimizer_state_dict' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
#         # 恢复步数
#         if 'global_step' in checkpoint:
#             global_step = checkpoint['global_step']
#             print(f"   -> Found Global Step: {global_step}")
            
#             # 计算我们需要跳过多少个物理 Batch
#             total_physical_batches = global_step * args.gradient_accumulation_steps
#             start_epoch = total_physical_batches // len(loader)
#             resume_batch_idx = total_physical_batches % len(loader)
            
#             print(f"   -> Resume Location: Epoch {start_epoch}, Batch {resume_batch_idx}")

#     # ====================================================
#     # 5. 训练循环 (Step-Based)
#     # ====================================================
#     model.train()
#     print(">>> Training Started... (BF16 Mode) <<<")
    
#     total_epochs = 999999 # 无限循环，由 max_train_steps 终止
    
#     for epoch in range(start_epoch, total_epochs):
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             # ⏩ 跳过已训练的数据 (精准续训)
#             if epoch == start_epoch and i < resume_batch_idx:
#                 if i % 50 == 0: print(f"⏩ Skipping batch {i}/{len(loader)}...", end='\r')
#                 continue

#             # --- 数据准备 ---
#             video = batch['video'].to(device, non_blocking=True)
#             state = batch['state'].to(device, non_blocking=True)
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
            
#             real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
#             real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            
#             # ==========================================================
#             # 🚨 Blind Masking 策略 (保留 80% Mask 逻辑)
#             # ==========================================================
#             mask_type_log = "Full_Input"
#             mask_prob = 0.8 
#             B = video.shape[0]
#             should_mask = torch.rand(B, device=device) < mask_prob
            
#             if should_mask.any():
#                 video[should_mask, 0] = 0.0
#                 ff[should_mask, 0] = 0.0 # 🚨 同步 Mask 首帧
#                 mask_type_log = "Masked_Main"
#             # ==========================================================

#             # Teacher 准备
#             siglip_target = torch.mean(real_siglip, dim=1)
#             exo_target = torch.mean(real_exo, dim=1)
            
#             noise_scale = 0.01 
#             siglip_target += torch.randn_like(siglip_target) * noise_scale
#             exo_target += torch.randn_like(exo_target) * noise_scale

#             teacher_feats = {
#                 "siglip_features": siglip_target,
#                 "exo_features": exo_target
#             }

#             # --- 前向传播 & Loss ---
#             with autocast('cuda', dtype=torch.bfloat16):
#                 out = model(video, text, state, ff)
                
#                 l_distill, _ = distill_fn(out, teacher_feats)
#                 l_decouple = decouple_fn(out['task_slots'], out['background_context'], out['task_confidence'])
#                 l_time = temporal_fn(out['temporal_head_output'])
                
#                 loss = loss_weights['distill'] * l_distill + \
#                        loss_weights['decouple'] * l_decouple + \
#                        loss_weights['consistency'] * l_time
                
#                 # 🌟 梯度累积归一化
#                 loss = loss / args.gradient_accumulation_steps

#             # --- 反向传播 ---
#             loss.backward()

#             # --- 参数更新 (每 accum steps 一次) ---
#             if (i + 1) % args.gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#                 optimizer.step()
#                 optimizer.zero_grad()
                
#                 global_step += 1
                
#                 # --- 日志记录 ---
#                 if global_step % 10 == 0:
#                     # 还原 loss 数值
#                     real_loss = loss.item() * args.gradient_accumulation_steps
                    
#                     elapsed = time.time() - start_time
#                     # 估算每个 step 的时间 (包含 accum)
#                     speed = (args.batch_size * args.gradient_accumulation_steps) / (elapsed / (i - resume_batch_idx + 1) * args.gradient_accumulation_steps + 1e-6)
                    
#                     print(f"Step [{global_step}/{args.max_train_steps}] Loss: {real_loss:.6f} | Distill: {l_distill.item():.6f} (Ep {epoch})")

#                     if args.use_wandb and HAS_WANDB:
#                         wandb.log({
#                             "total_loss": real_loss,
#                             "distill_loss": l_distill.item(),
#                             "decouple_loss": l_decouple.item(),
#                             "temporal_loss": l_time.item(),
#                             "global_step": global_step,
#                             "epoch": epoch,
#                             "lr": optimizer.param_groups[0]['lr']
#                         }, step=global_step)

#                 # --- 视频可视化 (每 500 步) ---
#                 if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
#                     try:
#                         vid_sample = video[0].float().cpu().numpy()
#                         main_view = np.transpose(vid_sample[0], (1, 0, 2, 3))
#                         wrist_view = np.transpose(vid_sample[1], (1, 0, 2, 3))
#                         combined_view = np.concatenate([main_view, wrist_view], axis=3)
                        
#                         wandb.log({
#                             "input_monitor": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"Step{global_step}: {mask_type_log}")
#                         }, step=global_step)
#                     except Exception as e:
#                         print(f"WandB upload failed: {e}")

#                 # --- Checkpoint 保存 ---
#                 if global_step % args.checkpointing_steps == 0:
#                     save_path = os.path.join(args.output_dir, f"16stageB_step_{global_step}.pt")
#                     torch.save({
#                         'epoch': epoch,
#                         'global_step': global_step,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                     }, save_path)
#                     print(f"💾 Checkpoint saved: {save_path}")

#                 # --- 结束训练 ---
#                 if global_step >= args.max_train_steps:
#                     print(f"🎉 Reached target {args.max_train_steps} steps. Training Finished.")
#                     final_path = os.path.join(args.output_dir, f"16stageB_final.pt")
#                     torch.save(model.state_dict(), final_path) # Final 只存权重方便加载
                    
#                     if args.use_wandb and HAS_WANDB: wandb.finish()
#                     return

#     if args.use_wandb and HAS_WANDB: wandb.finish()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, required=True) 
#     parser.add_argument('--stage_a_ckpt', type=str, default='/yanghaochuan/checkpoints/stageA_final.pt')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
    
#     # 训练超参
#     parser.add_argument('--batch_size', type=int, default=16) 
#     parser.add_argument('--num_workers', type=int, default=16)
    
#     # Step-based 控制参数
#     parser.add_argument('--max_train_steps', type=int, default=10000, help="Total training steps")
#     parser.add_argument('--checkpointing_steps', type=int, default=500, help="Save every X steps")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Simulate larger batch size")
    
#     # 杂项
#     parser.add_argument('--resume_from_checkpoint', type=str, default=None)
#     parser.add_argument('--use_wandb', action='store_true', default=False)
    
#     args = parser.parse_args()
    
#     train_stage_b(args)

# train/stageB_train.py
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

# === [环境检查] WandB ===
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️ WandB not found. Install with `pip install wandb`")

# 强制使用 Flash Attention
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.enable_mem_efficient_sdp(True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.fusion_encoder import FusionEncoder
from losses.distillation_loss import DistillationLoss
from losses.decoupling_regularizer import DecouplingLoss
from losses.temporal_consistency import TemporalConsistencyLoss
from utils.dataset_loader import RobotDataset

VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'

def train_stage_b(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage B Training: ForeSight Pre-training (World Model) ===")
    print(f"=== Mode: Step-Based | Target: {args.max_train_steps} Steps ===")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="ForeSight-StageB", 
            name=f"ForeSight_v2_{int(time.time())}",
            config=vars(args),
            resume="allow"
        )

    # 1. 初始化模型
    model = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
    if os.path.exists(args.stage_a_ckpt):
        print(f"Loading Stage A: {args.stage_a_ckpt}")
        model.load_state_dict(torch.load(args.stage_a_ckpt), strict=False)

    # 2. 参数冻结与解冻策略
    # 冻结 VideoMAE Backbone
    for param in model.backbone.parameters(): param.requires_grad = False
    
    # 解冻部分 Backbone 层
    layers_to_train = ["blocks.20", "blocks.21", "blocks.22", "blocks.23"] 
    count = 0
    for name, param in model.backbone.named_parameters():
        if any(x in name for x in layers_to_train) or "encoder.layer.2" in name: 
            param.requires_grad = True
            count += 1
    print(f"Unfrozen {count} parameters in VideoMAE backbone.")
    
    # 解冻所有非 Backbone 的层 (包括 Predictor, Heads, ViewEmbed 等)
    for name, param in model.named_parameters():
        if "backbone" not in name:
            param.requires_grad = True
    
    print("Compiling model with torch.compile...")
    # try: model = torch.compile(model)
    # except: pass

    # 3. 数据加载
    print(f"Loading data from: {args.data_root}")
    # 确保 dataset_loader 已更新 offsets=[0, 2, 4, 8, 16, 32]
    dataset = RobotDataset(hdf5_path=args.data_root, window_size=6) 
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    # 4. 损失与优化
    distill_fn = DistillationLoss()
    decouple_fn = DecouplingLoss()
    temporal_fn = TemporalConsistencyLoss()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # 权重调整
    loss_weights = {"distill": 1.0, "wm": 1.0, "decouple": 0.5, "consistency": 0.1}

    # 5. 断点续训逻辑
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"🔄 Resuming from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            total_physical_batches = global_step * args.gradient_accumulation_steps
            start_epoch = total_physical_batches // len(loader)
            resume_batch_idx = total_physical_batches % len(loader)

    # 6. 训练循环
    model.train()
    print(">>> Training Started... (BF16 Mode) <<<")
    
    total_epochs = 999999 
    
    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            if epoch == start_epoch and i < resume_batch_idx: continue

            # --- 数据搬运 ---
            video = batch['video'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            
            # Future Target [B, 6, 1152]
            future_exo_target = batch['future_exo_target'].to(device, non_blocking=True)

            # Semantic Teachers
            real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
            real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            
            # Masking 策略 (保留 Stage B 的 80% Masking)
            mask_type_log = "Full_Input"
            mask_prob = 1 
            B = video.shape[0]
            should_mask = torch.rand(B, device=device) < mask_prob
            if should_mask.any():
                video[should_mask, 0] = 0.0
                ff[should_mask, 0] = 0.0 
                mask_type_log = "Masked_Main"

            # 语义 Teacher 准备
            siglip_target = torch.mean(real_siglip, dim=1)
            exo_target = torch.mean(real_exo, dim=1) 
            
            if model.training:
                noise_scale = 0.01 
                siglip_target += torch.randn_like(siglip_target) * noise_scale
                exo_target += torch.randn_like(exo_target) * noise_scale

            teacher_feats = {"siglip_features": siglip_target, "exo_features": exo_target}

            # --- Forward & Loss ---
            with autocast('cuda', dtype=torch.bfloat16):
                out = model(video, text, state, ff)
                
                # 1. 语义蒸馏
                l_distill, _ = distill_fn(out, teacher_feats)
                
                # 2. 🟢 ForeSight WM Loss (MSE + Cosine)
                wm_pred = out['wm_latents'] # [B, 6, 1152]
                
                # MSE: 约束数值分布
                l_wm_mse = F.mse_loss(wm_pred, future_exo_target)
                
                # Cosine: 约束语义方向 (高维特征关键)
                wm_pred_norm = F.normalize(wm_pred, dim=-1)
                target_norm = F.normalize(future_exo_target, dim=-1)
                l_wm_cos = (1.0 - (wm_pred_norm * target_norm).sum(dim=-1)).mean()
                
                l_wm = l_wm_mse + 0.5 * l_wm_cos
                
                # 3. 其他正则
                l_decouple = decouple_fn(out['task_slots'], out['background_context'], out['task_confidence'])
                l_time = temporal_fn(out['temporal_head_output'])
                
                loss = loss_weights['distill'] * l_distill + \
                       loss_weights['wm'] * l_wm + \
                       loss_weights['decouple'] * l_decouple + \
                       loss_weights['consistency'] * l_time
                
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                if global_step % 10 == 0:
                    real_loss = loss.item() * args.gradient_accumulation_steps
                    print(f"Step {global_step} | L: {real_loss:.4f} | WM: {l_wm.item():.4f} (MSE:{l_wm_mse:.4f} Cos:{l_wm_cos:.4f})")

                    if args.use_wandb and HAS_WANDB:
                        wandb.log({
                            "total_loss": real_loss,
                            "wm_loss": l_wm.item(),
                            "wm_mse": l_wm_mse.item(),
                            "wm_cos": l_wm_cos.item(),
                            "distill_loss": l_distill.item(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "lr": optimizer.param_groups[0]['lr']
                        }, step=global_step)

                # Checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"124StageB_ForeSight_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                    print(f"💾 Checkpoint saved: {save_path}")

                if global_step >= args.max_train_steps:
                    print(f"🎉 Training Finished.")
                    final_path = os.path.join(args.output_dir, f"124StageB_ForeSight_final.pt")
                    torch.save(model.state_dict(), final_path)
                    if args.use_wandb and HAS_WANDB: wandb.finish()
                    return

    if args.use_wandb and HAS_WANDB: wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True) 
    parser.add_argument('--stage_a_ckpt', type=str, default='/yanghaochuan/checkpoints/stageA_final.pt')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
    parser.add_argument('--batch_size', type=int, default=16) 
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_train_steps', type=int, default=10000)
    parser.add_argument('--checkpointing_steps', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--resume_from_checkpoint', type=str, default='None')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    
    args = parser.parse_args()
    train_stage_b(args)