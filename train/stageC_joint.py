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
# from torch.utils.tensorboard import SummaryWriter 

# # === [环境检查] WandB ===
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

# # === 路径配置 (请确保这些路径正确) ===
# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'
# STATS_PATH = '/yanghaochuan/data/1223dataset_stats.json'

# def train_stage_c(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # ====================================================
#     # 0. 初始化日志
#     # ====================================================
#     log_dir = os.path.join(args.output_dir, "logs")
#     os.makedirs(log_dir, exist_ok=True)
#     tb_writer = SummaryWriter(log_dir=log_dir)
    
#     if args.use_wandb and HAS_WANDB:
#         wandb.init(
#             project="RDT-StageC-Joint",
#             # 名字里带上 step 和 acc，方便区分实验
#             name=f"step{args.max_train_steps}_acc{args.gradient_accumulation_steps}_{int(time.time())}",
#             config=vars(args),
#             resume="allow"
#         )
    
#     print(f"=== Stage C Joint Training (Step-Based) ===")
#     print(f"🎯 Target: {args.max_train_steps} Global Steps")
#     print(f"📦 Physical Batch Size: {args.batch_size}")
#     print(f"🔋 Gradient Accumulation: {args.gradient_accumulation_steps}")
#     print(f"🔥 Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")

#     # ====================================================
#     # 1. 模型加载
#     # ====================================================
#     print("Loading Models...")
#     fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
#     # 加载 Stage B (如果有)
#     if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B: {args.stage_b_ckpt}")
#         ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
#         state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
#         fusion_encoder.load_state_dict(state_dict, strict=False)
    
#     # 冻结 VideoMAE，只微调 Adapter
#     fusion_encoder.eval() 
#     for param in fusion_encoder.parameters(): param.requires_grad = True 
#     for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
#     if fusion_encoder.text_encoder:
#         for p in fusion_encoder.text_encoder.parameters(): p.requires_grad = False

#     # 加载 RDT
#     rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, pred_horizon=args.pred_horizon).to(device)
    
#     # RDT 权重切片加载逻辑 (保留你之前的逻辑)
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

#     # LoRA 配置
#     print("Applying LoRA...")
#     peft_config = LoraConfig(
#         r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
#         lora_dropout=0.05, bias="none"
#     )
#     rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    
#     # ====================================================
#     # 2. 优化器 & 调度器
#     # ====================================================
#     params = [
#         {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
#         {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
#     ]
#     optimizer = optim.AdamW(params, weight_decay=1e-4)
    
#     # ⚠️ 训练时保持 1000 步，不要改这里！推理才用 25 步。
#     noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

#     # ====================================================
#     # 3. 数据加载
#     # ====================================================
#     print(f"Loading Dataset from {args.data_root}")
#     dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, pred_horizon=args.pred_horizon, stats_path=STATS_PATH)
#     # drop_last=True 防止最后一个 batch 只有几个样本导致梯度不稳定
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

#     # ====================================================
#     # 4. 精准断点续训逻辑
#     # ====================================================
#     global_step = 0
#     start_epoch = 0
#     resume_batch_idx = 0

#     if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
#         print(f"🔄 Resuming from: {args.resume_from_checkpoint}")
#         checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
#         # 恢复权重
#         if 'rdt_state_dict' in checkpoint: rdt_wrapper.load_state_dict(checkpoint['rdt_state_dict'], strict=False)
#         if 'encoder_state_dict' in checkpoint: fusion_encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
#         if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
#         # 恢复步数
#         if 'global_step' in checkpoint:
#             global_step = checkpoint['global_step']
#             print(f"   -> Found Global Step: {global_step}")
            
#             # 计算我们需要跳过多少个物理 Batch
#             # 公式：总物理Batch数 = Global Step * 梯度累积次数
#             total_physical_batches = global_step * args.gradient_accumulation_steps
            
#             # 计算当前 Epoch 和 Batch 索引
#             start_epoch = total_physical_batches // len(loader)
#             resume_batch_idx = total_physical_batches % len(loader)
            
#             print(f"   -> Resume Location: Epoch {start_epoch}, Batch {resume_batch_idx}")
#             print(f"   -> Ready to continue towards {args.max_train_steps} steps.")
        
#         elif 'epoch' in checkpoint:
#             # 兼容旧逻辑
#             start_epoch = checkpoint['epoch'] + 1
#             global_step = start_epoch * len(loader) // args.gradient_accumulation_steps
#             print(f"   -> Resuming from Epoch {start_epoch} (Approx. Step {global_step})")

#     # ====================================================
#     # 5. 训练循环 (16x4=64 Logic)
#     # ====================================================
#     print(">>> Training Started <<<")
    
#     # 这是一个足够大的数字，确保循环只由 max_train_steps 终止
#     total_epochs = 999999 
    
#     for epoch in range(start_epoch, total_epochs):
#         rdt_wrapper.train()
        
#         for i, batch in enumerate(loader):
#             # ⏩ 跳过已训练的数据 (精确续训)
#             if epoch == start_epoch and i < resume_batch_idx:
#                 if i % 50 == 0: print(f"⏩ Skipping batch {i}/{len(loader)}...", end='\r')
#                 continue

#             # --- 数据准备 ---
#             video = batch['video'].to(device, non_blocking=True)
#             state = batch['state'].to(device, non_blocking=True)
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
#             actions = batch['action_target'].to(device, non_blocking=True)

#             # Modality Dropout
#             rand_val = torch.rand(1).item()
#             mask_type = "None"
#             if rand_val < 0.7: 
#                 video[:, 0] = 0.0
#                 mask_type = "Main_Masked"
#             elif rand_val < 0.8: 
#                 video[:, 1] = 0.0
#                 mask_type = "Wrist_Masked"
            
#             # --- 前向传播 ---
#             with autocast('cuda', dtype=torch.bfloat16):
#                 # 1. Encode
#                 e_t = fusion_encoder(video, text, state, ff)['e_t']
                
#                 # 2. Noise & Target
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 # 3. Predict
#                 conditions = {"e_t": e_t, "state": state[:, -1, :]}
#                 pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
#                 # 4. Loss Calculation
#                 loss_diff = F.mse_loss(pred_noise, noise)
#                 loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
#                 total_loss = loss_diff + 0.1 * loss_cons
                
#                 # 🌟 [关键点 1] 梯度累积归一化
#                 # 因为 backward 会累加梯度，所以 Loss 要除以累积步数，保证平均梯度幅度不变
#                 total_loss = total_loss / args.gradient_accumulation_steps

#             # --- 反向传播 ---
#             total_loss.backward()

#             # 🌟 [关键点 2] 参数更新逻辑
#             # 只有当累积了足够的步数，或者 Epoch 结束时，才更新参数
#             if (i + 1) % args.gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
#                 optimizer.step()
#                 optimizer.zero_grad()
                
#                 # 真正的 Step 增加了 (Update Step)
#                 global_step += 1
                
#                 # --- 日志记录 (还原 Loss 数值方便观察) ---
#                 if global_step % 10 == 0:
#                     real_loss = total_loss.item() * args.gradient_accumulation_steps
#                     real_diff = loss_diff.item()
#                     real_cons = loss_cons.item()
                    
#                     print(f"Step [{global_step}/{args.max_train_steps}] Loss: {real_loss:.4f} (Ep {epoch})")
                    
#                     tb_writer.add_scalar('Train/Total_Loss', real_loss, global_step)
#                     if args.use_wandb and HAS_WANDB:
#                         wandb.log({
#                             "total_loss": real_loss,
#                             "diff_loss": real_diff,
#                             "cons_loss": real_cons,
#                             "global_step": global_step,
#                             "epoch": epoch
#                         }, step=global_step)

#                 # --- 视频可视化 (每 500 步) ---
#                 if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
#                     try:
#                         vid_sample = video[0].float().cpu().numpy() 
#                         main_view = np.transpose(vid_sample[0], (1, 0, 2, 3))
#                         wrist_view = np.transpose(vid_sample[1], (1, 0, 2, 3))
#                         combined_view = np.concatenate([main_view, wrist_view], axis=3) 
#                         wandb.log({
#                             "input_video": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"S{global_step}: {mask_type}")
#                         }, step=global_step)
#                     except: pass

#                 # --- 💾 阶段性保存 (Checkpointing) ---
#                 if global_step % args.checkpointing_steps == 0:
#                     save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}.pt")
#                     torch.save({
#                         'epoch': epoch,
#                         'global_step': global_step, # 保存当前 Global Step
#                         'rdt_state_dict': rdt_wrapper.state_dict(),
#                         'encoder_state_dict': fusion_encoder.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'pred_horizon': args.pred_horizon
#                     }, save_path)
#                     print(f"💾 Checkpoint saved: {save_path}")

#                 # --- 🛑 停止训练 ---
#                 if global_step >= args.max_train_steps:
#                     print(f"🎉 Reached target {args.max_train_steps} steps. Training Finished.")
#                     # 保存最终模型
#                     final_path = os.path.join(args.output_dir, f"checkpoint_final_{global_step}.pt")
#                     torch.save({
#                         'epoch': epoch,
#                         'global_step': global_step,
#                         'rdt_state_dict': rdt_wrapper.state_dict(),
#                         'encoder_state_dict': fusion_encoder.state_dict()
#                     }, final_path)
#                     tb_writer.close()
#                     if args.use_wandb and HAS_WANDB: wandb.finish()
#                     return # 退出函数，结束脚本

#     tb_writer.close()
#     if args.use_wandb and HAS_WANDB: wandb.finish()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/1225checkpoints')
#     parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/checkpoints/1223stageB_papercup.pt')
    
#     # 物理 Batch Size (显存限制，保持 16)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--pred_horizon', type=int, default=64)
    
#     # === 关键控制参数 ===
#     # 梯度累积：设为 4，使得 Effective Batch Size = 16 * 4 = 64
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
#                         help="Number of updates steps to accumulate before update pass. (Effective BS = batch_size * this)")
    
#     # 目标总步数 (Update Steps)
#     parser.add_argument('--max_train_steps', type=int, default=10000, 
#                         help="Total number of training steps (parameter updates) to perform.")
    
#     # 每多少步保存一次
#     parser.add_argument('--checkpointing_steps', type=int, default=500, 
#                         help="Save checkpoint every X updates.")
    
#     # 续训
#     parser.add_argument('--resume_from_checkpoint', type=str, default=None)
#     parser.add_argument('--use_wandb', action='store_true', default=False)
    
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

# === [环境检查] WandB ===
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
# 🚨 [新增] 引入蒸馏 Loss 作为正则项
from losses.distillation_loss import DistillationLoss

# === 路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = '/yanghaochuan/data/1223dataset_stats.json'

def train_stage_c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ====================================================
    # 0. 初始化日志
    # ====================================================
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="RDT-StageC-Joint",
            name=f"step{args.max_train_steps}_acc{args.gradient_accumulation_steps}_{int(time.time())}",
            config=vars(args),
            resume="allow"
        )
    
    print(f"=== Stage C Joint Training (Step-Based + Regularization) ===")
    print(f"🎯 Target: {args.max_train_steps} Global Steps")
    print(f"📦 Physical Batch Size: {args.batch_size}")
    print(f"🔋 Gradient Accumulation: {args.gradient_accumulation_steps}")

    # ====================================================
    # 1. 模型加载
    # ====================================================
    print("Loading Models...")
    fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
    # 加载 Stage B
    if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
        print(f"Loading Stage B: {args.stage_b_ckpt}")
        ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
        state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
        # 兼容 key
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."): new_state_dict[k[7:]] = v
            else: new_state_dict[k] = v
        fusion_encoder.load_state_dict(new_state_dict, strict=False)
    
    # 冻结 VideoMAE，只微调 Adapter
    fusion_encoder.eval() 
    for param in fusion_encoder.parameters(): param.requires_grad = True 
    for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
    if fusion_encoder.text_encoder:
        for p in fusion_encoder.text_encoder.parameters(): p.requires_grad = False

    # 加载 RDT
    rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, pred_horizon=args.pred_horizon).to(device)
    
    # RDT 权重切片加载逻辑
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
    # 2. 优化器 & Loss
    # ====================================================
    params = [
        {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

    # 🚨 [新增] 蒸馏 Loss (用于正则化)
    distill_fn = DistillationLoss()

    # ====================================================
    # 3. 数据加载
    # ====================================================
    print(f"Loading Dataset from {args.data_root}")
    dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, pred_horizon=args.pred_horizon, stats_path=STATS_PATH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # ====================================================
    # 4. 精准断点续训
    # ====================================================
    global_step = 0
    start_epoch = 0
    resume_batch_idx = 0

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"🔄 Resuming from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
        if 'rdt_state_dict' in checkpoint: rdt_wrapper.load_state_dict(checkpoint['rdt_state_dict'], strict=False)
        if 'encoder_state_dict' in checkpoint: fusion_encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
            total_physical_batches = global_step * args.gradient_accumulation_steps
            start_epoch = total_physical_batches // len(loader)
            resume_batch_idx = total_physical_batches % len(loader)
            print(f"   -> Resume Location: Epoch {start_epoch}, Batch {resume_batch_idx}")
        elif 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            global_step = start_epoch * len(loader) // args.gradient_accumulation_steps
            print(f"   -> Resuming from Epoch {start_epoch} (Approx. Step {global_step})")

    # ====================================================
    # 5. 训练循环
    # ====================================================
    print(">>> Training Started <<<")
    total_epochs = 999999 
    
    for epoch in range(start_epoch, total_epochs):
        rdt_wrapper.train()
        
        for i, batch in enumerate(loader):
            if epoch == start_epoch and i < resume_batch_idx:
                if i % 50 == 0: print(f"⏩ Skipping batch {i}/{len(loader)}...", end='\r')
                continue

            # --- 数据准备 ---
            video = batch['video'].to(device, non_blocking=True) # [B, 2, C, T, H, W]
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            actions = batch['action_target'].to(device, non_blocking=True)

            # --- Teacher Features (用于正则化) ---
            real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
            real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            siglip_target = torch.mean(real_siglip, dim=1)
            exo_target = torch.mean(real_exo, dim=1)
            teacher_feats = {"siglip_features": siglip_target, "exo_features": exo_target}

            # ==========================================================
            # 🚨 [关键修复] Modality Dropout (数据克隆与同步 Mask)
            # ==========================================================
            rand_val = torch.rand(1).item()
            mask_type = "Teacher_Mode"
            
            # 1. 必须 Clone! 否则会污染 batch['video']，导致 Consistency Loss 里的 Teacher 也是黑的
            video_input = video.clone()
            ff_input = ff.clone()
            
            if rand_val < 0.7: 
                # 70% 概率：模拟推理 (Student Mode)
                video_input[:, 0] = 0.0
                ff_input[:, 0] = 0.0     # <--- 🚨 同步 Mask 首帧 (防作弊)
                mask_type = "Main_Masked"
            elif rand_val < 0.8: 
                # 10% 概率：手腕遮挡
                video_input[:, 1] = 0.0
                ff_input[:, 1] = 0.0     # <--- 🚨 同步 Mask 首帧
                mask_type = "Wrist_Masked"
            # 20% 概率：Teacher Mode (全可见)
            
            # ==========================================================
            
            # --- 前向传播 ---
            with autocast('cuda', dtype=torch.bfloat16):
                # 1. Encode (使用 Mask 后的输入)
                # 返回完整 dict 以便计算 Distill Loss
                encoder_out = fusion_encoder(video_input, text, state, ff_input)
                e_t = encoder_out['e_t']
                
                # 2. RDT Forward (Action Loss)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                conditions = {"e_t": e_t, "state": state[:, -1, :]}
                pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
                # --- Loss Calculation ---
                
                # Loss 1: Action Diffusion Loss
                loss_diff = F.mse_loss(pred_noise, noise)
                
                # Loss 2: Consistency Loss (Brain Completion)
                # 使用原始 batch (包含未 Mask 的数据)，函数内部会自己处理 Student/Teacher 构建
                loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                
                # Loss 3: Distillation Regularization (Don't forget semantics)
                # 强迫当前 Mask 状态下的 encoder_out 依然能恢复出全局语义
                # 这完全复用了 Stage B 的逻辑
                loss_distill_reg, _ = distill_fn(encoder_out, teacher_feats)
                
                # 🌟 组合 Loss
                # diff: 1.0 (主任务)
                # cons: 0.1 (辅助脑补)
                # distill: 0.05 (辅助语义锚定，防止漂移)
                total_loss = loss_diff + 0.1 * loss_cons + 0.05 * loss_distill_reg
                
                # 梯度累积归一化
                total_loss = total_loss / args.gradient_accumulation_steps

            # --- 反向传播 ---
            total_loss.backward()

            # --- 参数更新 ---
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # --- 日志记录 ---
                if global_step % 10 == 0:
                    real_loss = total_loss.item() * args.gradient_accumulation_steps
                    
                    print(f"Step [{global_step}/{args.max_train_steps}] Loss: {real_loss:.4f} | Diff: {loss_diff.item():.4f} | Cons: {loss_cons.item():.4f} | Reg: {loss_distill_reg.item():.4f}")
                    
                    tb_writer.add_scalar('Train/Total_Loss', real_loss, global_step)
                    if args.use_wandb and HAS_WANDB:
                        wandb.log({
                            "total_loss": real_loss,
                            "diff_loss": loss_diff.item(),
                            "cons_loss": loss_cons.item(),
                            "distill_reg_loss": loss_distill_reg.item(),
                            "global_step": global_step,
                            "epoch": epoch,
                            "lr": optimizer.param_groups[0]['lr']
                        }, step=global_step)

                # --- 视频可视化 ---
                if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
                    try:
                        # 可视化真正喂给 RDT 的数据 (video_input)
                        vid_sample = video_input[0].float().cpu().numpy() 
                        main_view = np.transpose(vid_sample[0], (1, 0, 2, 3))
                        wrist_view = np.transpose(vid_sample[1], (1, 0, 2, 3))
                        combined_view = np.concatenate([main_view, wrist_view], axis=3) 
                        wandb.log({
                            "input_monitor": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"S{global_step}: {mask_type}")
                        }, step=global_step)
                    except: pass

                # --- Checkpoint 保存 ---
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"stageC_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step, 
                        'rdt_state_dict': rdt_wrapper.state_dict(),
                        'encoder_state_dict': fusion_encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'pred_horizon': args.pred_horizon
                    }, save_path)
                    print(f"💾 Checkpoint saved: {save_path}")

                # --- 结束训练 ---
                if global_step >= args.max_train_steps:
                    print(f"🎉 Reached target {args.max_train_steps} steps. Training Finished.")
                    final_path = os.path.join(args.output_dir, f"stageC_final_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'rdt_state_dict': rdt_wrapper.state_dict(),
                        'encoder_state_dict': fusion_encoder.state_dict()
                    }, final_path)
                    tb_writer.close()
                    if args.use_wandb and HAS_WANDB: wandb.finish()
                    return 

    tb_writer.close()
    if args.use_wandb and HAS_WANDB: wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/yanghaochuan/data/1223pick_up_the_paper_cup.hdf5')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/1225checkpoints')
    parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/checkpoints/1223stageB_papercup.pt')
    
    # 物理 Batch Size (显存限制，保持 16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pred_horizon', type=int, default=64)
    
    # === 关键控制参数 ===
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, 
                        help="Number of updates steps to accumulate before update pass. (Effective BS = batch_size * this)")
    
    parser.add_argument('--max_train_steps', type=int, default=10000, 
                        help="Total number of training steps (parameter updates) to perform.")
    
    parser.add_argument('--checkpointing_steps', type=int, default=500, 
                        help="Save checkpoint every X updates.")
    
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    
    args = parser.parse_args()
    train_stage_c(args)