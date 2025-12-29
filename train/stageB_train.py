# # train/stageB_train_fast.py
# import sys
# import os
# import torch
# import torch.optim as optim
# import argparse
# import time
# from torch.utils.data import DataLoader
# from torch.amp import autocast # BF16 不需要 GradScaler

# # 强制使用 Flash Attention，禁用普通数学注意力
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
#     print(f"=== Mode: BF16 (Fast & Stable) | Workers: {args.num_workers} | Batch: {args.batch_size} ===")
    
#     # 启用 TF32 (在 A800 上能加速 FP32/BF16 计算)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     # 1. 初始化模型
#     model = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
#     if os.path.exists(args.stage_a_ckpt):
#         print(f"Loading Stage A: {args.stage_a_ckpt}")
#         model.load_state_dict(torch.load(args.stage_a_ckpt), strict=False)

#     # 冻结参数
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
    
#     # === 优化点 A: 编译模型 (PyTorch 2.0+) ===
#     print("Compiling model with torch.compile... (First step will be slow)")
#     # try:
#     #     model = torch.compile(model)
#     # except Exception as e:
#     #     print(f"Compile failed, falling back to eager mode: {e}")

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
    
#     # 学习率可以适当回升，因为 BF16 很稳，且 Batch Size 变大了
#     optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
#     loss_weights = {"distill":1.0, "decouple":0.5, "consistency":1.0}

#     # 4. 训练循环
#     model.train()
#     print(">>> Training Started... (BF16 Mode) <<<")
    
#     for epoch in range(args.epochs):
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             video = batch['video'].to(device, non_blocking=True)
#             state = batch['state'].to(device, non_blocking=True)
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
            
#             real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
#             real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            
#             # 时序平均
#             siglip_target = torch.mean(real_siglip, dim=1)
#             exo_target = torch.mean(real_exo, dim=1)
            
#             # === 新增：给 Teacher 特征加噪声 (仅在训练时) ===
#             if model.training:
#                 noise_scale = 0.01 # 视特征数值范围而定，通常 0.01-0.05
#                 siglip_target += torch.randn_like(siglip_target) * noise_scale
#                 exo_target += torch.randn_like(exo_target) * noise_scale
#             # ============================================

#             teacher_feats = {
#                 "siglip_features": siglip_target,
#                 "exo_features": exo_target
#             }

#             optimizer.zero_grad()

#             # === 优化点 B: 启用 BFloat16 ===
#             # A800 专属：不需要 Scaler，因为范围够大，不会溢出
#             with autocast('cuda', dtype=torch.bfloat16):
#                 out = model(video, text, state, ff)
                
#                 l_distill, _ = distill_fn(out, teacher_feats)
#                 l_decouple = decouple_fn(out['task_slots'], out['background_context'], out['task_confidence'])
#                 l_time = temporal_fn(out['temporal_head_output'])
                
#                 loss = loss_weights['distill'] * l_distill + \
#                        loss_weights['decouple'] * l_decouple + \
#                        loss_weights['consistency'] * l_time

#             # === 普通反向传播 (BF16 不需要 scaler.step) ===
#             loss.backward()
            
#             # 依然保留梯度裁剪以防万一
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             # 进度打印
#             if i % 10 == 0 and i > 0:
#                 elapsed = time.time() - start_time
#                 speed = (i * args.batch_size) / elapsed
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.8e} | Distill: {l_distill.item():.8e} | Speed: {speed:.1f} img/s")

#     os.makedirs(args.output_dir, exist_ok=True)
#     save_path = os.path.join(args.output_dir, "1223stageB_papercup.pt")
#     torch.save(model.state_dict(), save_path)
#     print(f"Saved to {save_path}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, required=True) 
#     parser.add_argument('--stage_a_ckpt', type=str, default='/yanghaochuan/checkpoints/stageA_final.pt')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
#     # === 优化点 C: Batch Size 翻倍 ===
#     parser.add_argument('--batch_size', type=int, default=24) # 直接上 48！
#     parser.add_argument('--num_workers', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=2)
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
from torch.utils.data import DataLoader
from torch.amp import autocast # BF16 不需要 GradScaler

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
    print(f"=== Stage B Training: Real Teacher Distillation on {device} ===")
    print(f"=== Mode: Step-Based Training | Target: {args.max_train_steps} Steps ===")
    
    # 启用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # === WandB 初始化 ===
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="RDT-StageB-Pretrain",
            name=f"step{args.max_train_steps}_mask0.8_{int(time.time())}",
            config=vars(args),
            resume="allow"
        )

    # 1. 初始化模型
    model = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
    if os.path.exists(args.stage_a_ckpt):
        print(f"Loading Stage A: {args.stage_a_ckpt}")
        model.load_state_dict(torch.load(args.stage_a_ckpt), strict=False)

    # 冻结 VideoMAE，只微调 Adapter 和部分 Block
    for param in model.backbone.parameters(): param.requires_grad = False
    
    layers_to_train = ["blocks.20", "blocks.21", "blocks.22", "blocks.23"] 
    count = 0
    for name, param in model.backbone.named_parameters():
        if any(x in name for x in layers_to_train) or "encoder.layer.2" in name: 
            param.requires_grad = True
            count += 1
    print(f"Unfrozen {count} parameters in VideoMAE backbone.")
    
    for p in model.routing_layer.parameters(): p.requires_grad = True
    for p in model.semantic_align_head.parameters(): p.requires_grad = True
    for p in model.temporal_align_head.parameters(): p.requires_grad = True
    for p in model.projection_head.parameters(): p.requires_grad = True
    
    print("Compiling model with torch.compile...")
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(f"Compile failed: {e}")

    # 2. 数据加载
    print(f"Loading data from: {args.data_root}")
    dataset = RobotDataset(hdf5_path=args.data_root, window_size=16)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    # 3. 损失与优化
    distill_fn = DistillationLoss()
    decouple_fn = DecouplingLoss()
    temporal_fn = TemporalConsistencyLoss()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    loss_weights = {"distill":1.0, "decouple":0.5, "consistency":1.0}

    # ====================================================
    # 4. 精准断点续训逻辑 (Step-Based Resume)
    # ====================================================
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"🔄 Resuming from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
        # 恢复权重
        # 兼容只保存了 model state dict 的情况，也兼容保存了完整 info 的情况
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) # 旧格式

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复步数
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            print(f"   -> Found Global Step: {global_step}")
            
            # 计算我们需要跳过多少个物理 Batch
            total_physical_batches = global_step * args.gradient_accumulation_steps
            start_epoch = total_physical_batches // len(loader)
            resume_batch_idx = total_physical_batches % len(loader)
            
            print(f"   -> Resume Location: Epoch {start_epoch}, Batch {resume_batch_idx}")

    # ====================================================
    # 5. 训练循环 (Step-Based)
    # ====================================================
    model.train()
    print(">>> Training Started... (BF16 Mode) <<<")
    
    total_epochs = 999999 # 无限循环，由 max_train_steps 终止
    
    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            # ⏩ 跳过已训练的数据 (精准续训)
            if epoch == start_epoch and i < resume_batch_idx:
                if i % 50 == 0: print(f"⏩ Skipping batch {i}/{len(loader)}...", end='\r')
                continue

            # --- 数据准备 ---
            video = batch['video'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            
            real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
            real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            
            # ==========================================================
            # 🚨 Blind Masking 策略 (保留 80% Mask 逻辑)
            # ==========================================================
            mask_type_log = "Full_Input"
            mask_prob = 0.8 
            B = video.shape[0]
            should_mask = torch.rand(B, device=device) < mask_prob
            
            if should_mask.any():
                video[should_mask, 0] = 0.0
                ff[should_mask, 0] = 0.0 # 🚨 同步 Mask 首帧
                mask_type_log = "Masked_Main"
            # ==========================================================

            # Teacher 准备
            siglip_target = torch.mean(real_siglip, dim=1)
            exo_target = torch.mean(real_exo, dim=1)
            
            noise_scale = 0.01 
            siglip_target += torch.randn_like(siglip_target) * noise_scale
            exo_target += torch.randn_like(exo_target) * noise_scale

            teacher_feats = {
                "siglip_features": siglip_target,
                "exo_features": exo_target
            }

            # --- 前向传播 & Loss ---
            with autocast('cuda', dtype=torch.bfloat16):
                out = model(video, text, state, ff)
                
                l_distill, _ = distill_fn(out, teacher_feats)
                l_decouple = decouple_fn(out['task_slots'], out['background_context'], out['task_confidence'])
                l_time = temporal_fn(out['temporal_head_output'])
                
                loss = loss_weights['distill'] * l_distill + \
                       loss_weights['decouple'] * l_decouple + \
                       loss_weights['consistency'] * l_time
                
                # 🌟 梯度累积归一化
                loss = loss / args.gradient_accumulation_steps

            # --- 反向传播 ---
            loss.backward()

            # --- 参数更新 (每 accum steps 一次) ---
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # --- 日志记录 ---
                if global_step % 10 == 0:
                    # 还原 loss 数值
                    real_loss = loss.item() * args.gradient_accumulation_steps
                    
                    elapsed = time.time() - start_time
                    # 估算每个 step 的时间 (包含 accum)
                    speed = (args.batch_size * args.gradient_accumulation_steps) / (elapsed / (i - resume_batch_idx + 1) * args.gradient_accumulation_steps + 1e-6)
                    
                    print(f"Step [{global_step}/{args.max_train_steps}] Loss: {real_loss:.6f} | Distill: {l_distill.item():.6f} (Ep {epoch})")

                    if args.use_wandb and HAS_WANDB:
                        wandb.log({
                            "total_loss": real_loss,
                            "distill_loss": l_distill.item(),
                            "decouple_loss": l_decouple.item(),
                            "temporal_loss": l_time.item(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "lr": optimizer.param_groups[0]['lr']
                        }, step=global_step)

                # --- 视频可视化 (每 500 步) ---
                if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
                    try:
                        vid_sample = video[0].float().cpu().numpy()
                        main_view = np.transpose(vid_sample[0], (1, 0, 2, 3))
                        wrist_view = np.transpose(vid_sample[1], (1, 0, 2, 3))
                        combined_view = np.concatenate([main_view, wrist_view], axis=3)
                        
                        wandb.log({
                            "input_monitor": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"Step{global_step}: {mask_type_log}")
                        }, step=global_step)
                    except Exception as e:
                        print(f"WandB upload failed: {e}")

                # --- Checkpoint 保存 ---
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"1226stageB_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)
                    print(f"💾 Checkpoint saved: {save_path}")

                # --- 结束训练 ---
                if global_step >= args.max_train_steps:
                    print(f"🎉 Reached target {args.max_train_steps} steps. Training Finished.")
                    final_path = os.path.join(args.output_dir, f"stageB_final.pt")
                    torch.save(model.state_dict(), final_path) # Final 只存权重方便加载
                    
                    if args.use_wandb and HAS_WANDB: wandb.finish()
                    return

    if args.use_wandb and HAS_WANDB: wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True) 
    parser.add_argument('--stage_a_ckpt', type=str, default='/yanghaochuan/checkpoints/stageA_final.pt')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
    
    # 训练超参
    parser.add_argument('--batch_size', type=int, default=24) 
    parser.add_argument('--num_workers', type=int, default=16)
    
    # Step-based 控制参数
    parser.add_argument('--max_train_steps', type=int, default=20000, help="Total training steps")
    parser.add_argument('--checkpointing_steps', type=int, default=2000, help="Save every X steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Simulate larger batch size")
    
    # 杂项
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    
    args = parser.parse_args()
    
    train_stage_b(args)