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
# # 🚨 [新增] 引入蒸馏 Loss 作为正则项
# from losses.distillation_loss import DistillationLoss

# # === 路径配置 ===
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
#             name=f"step{args.max_train_steps}_acc{args.gradient_accumulation_steps}_{int(time.time())}",
#             config=vars(args),
#             resume="allow"
#         )
    
#     print(f"=== Stage C Joint Training (Step-Based + Regularization) ===")
#     print(f"🎯 Target: {args.max_train_steps} Global Steps")
#     print(f"📦 Physical Batch Size: {args.batch_size}")
#     print(f"🔋 Gradient Accumulation: {args.gradient_accumulation_steps}")

#     # ====================================================
#     # 1. 模型加载
#     # ====================================================
#     print("Loading Models...")
#     fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
#     # 加载 Stage B
#     if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B: {args.stage_b_ckpt}")
#         ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
#         state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
#         # 兼容 key
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if k.startswith("module."): new_state_dict[k[7:]] = v
#             else: new_state_dict[k] = v
#         fusion_encoder.load_state_dict(new_state_dict, strict=False)
    
#     # 冻结 VideoMAE，只微调 Adapter
#     fusion_encoder.eval() 
#     for param in fusion_encoder.parameters(): param.requires_grad = True 
#     for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
#     if fusion_encoder.text_encoder:
#         for p in fusion_encoder.text_encoder.parameters(): p.requires_grad = False

#     # 加载 RDT
#     rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, pred_horizon=args.pred_horizon).to(device)
    
#     # RDT 权重切片加载逻辑
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
#     # 2. 优化器 & Loss
#     # ====================================================
#     params = [
#         {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
#         {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
#     ]
#     optimizer = optim.AdamW(params, weight_decay=1e-4)
    
#     noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")

#     # 🚨 [新增] 蒸馏 Loss (用于正则化)
#     distill_fn = DistillationLoss()

#     # ====================================================
#     # 3. 数据加载
#     # ====================================================
#     print(f"Loading Dataset from {args.data_root}")
#     dataset = RobotDataset(hdf5_path=args.data_root, window_size=16, pred_horizon=args.pred_horizon, stats_path=STATS_PATH)
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

#     # ====================================================
#     # 4. 精准断点续训
#     # ====================================================
#     global_step = 0
#     start_epoch = 0
#     resume_batch_idx = 0

#     if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
#         print(f"🔄 Resuming from: {args.resume_from_checkpoint}")
#         checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        
#         if 'rdt_state_dict' in checkpoint: rdt_wrapper.load_state_dict(checkpoint['rdt_state_dict'], strict=False)
#         if 'encoder_state_dict' in checkpoint: fusion_encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
#         if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
#         if 'global_step' in checkpoint:
#             global_step = checkpoint['global_step']
#             total_physical_batches = global_step * args.gradient_accumulation_steps
#             start_epoch = total_physical_batches // len(loader)
#             resume_batch_idx = total_physical_batches % len(loader)
#             print(f"   -> Resume Location: Epoch {start_epoch}, Batch {resume_batch_idx}")
#         elif 'epoch' in checkpoint:
#             start_epoch = checkpoint['epoch'] + 1
#             global_step = start_epoch * len(loader) // args.gradient_accumulation_steps
#             print(f"   -> Resuming from Epoch {start_epoch} (Approx. Step {global_step})")

#     # ====================================================
#     # 5. 训练循环
#     # ====================================================
#     print(">>> Training Started <<<")
#     total_epochs = 999999 
    
#     for epoch in range(start_epoch, total_epochs):
#         rdt_wrapper.train()
        
#         for i, batch in enumerate(loader):
#             if epoch == start_epoch and i < resume_batch_idx:
#                 if i % 50 == 0: print(f"⏩ Skipping batch {i}/{len(loader)}...", end='\r')
#                 continue

#             # --- 数据准备 ---
#             video = batch['video'].to(device, non_blocking=True) # [B, 2, C, T, H, W]
#             state = batch['state'].to(device, non_blocking=True)
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
#             actions = batch['action_target'].to(device, non_blocking=True)

#             # --- Teacher Features (用于正则化) ---
#             real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
#             real_exo = batch['teacher_exo'].to(device, non_blocking=True)
#             siglip_target = torch.mean(real_siglip, dim=1)
#             exo_target = torch.mean(real_exo, dim=1)
#             teacher_feats = {"siglip_features": siglip_target, "exo_features": exo_target}

#             # ==========================================================
#             # 🚨 [关键修复] Modality Dropout (数据克隆与同步 Mask)
#             # ==========================================================
#             rand_val = torch.rand(1).item()
#             mask_type = "Teacher_Mode"
            
#             # 1. 必须 Clone! 否则会污染 batch['video']，导致 Consistency Loss 里的 Teacher 也是黑的
#             video_input = video.clone()
#             ff_input = ff.clone()
            
#             if rand_val < 0.7: 
#                 # 70% 概率：模拟推理 (Student Mode)
#                 video_input[:, 0] = 0.0
#                 ff_input[:, 0] = 0.0     # <--- 🚨 同步 Mask 首帧 (防作弊)
#                 mask_type = "Main_Masked"
#             elif rand_val < 0.8: 
#                 # 10% 概率：手腕遮挡
#                 video_input[:, 1] = 0.0
#                 ff_input[:, 1] = 0.0     # <--- 🚨 同步 Mask 首帧
#                 mask_type = "Wrist_Masked"
#             # 20% 概率：Teacher Mode (全可见)
            
#             # ==========================================================
            
#             # --- 前向传播 ---
#             with autocast('cuda', dtype=torch.bfloat16):
#                 # 1. Encode (使用 Mask 后的输入)
#                 # 返回完整 dict 以便计算 Distill Loss
#                 encoder_out = fusion_encoder(video_input, text, state, ff_input)
#                 e_t = encoder_out['e_t']
                
#                 # 2. RDT Forward (Action Loss)
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 conditions = {"e_t": e_t, "state": state[:, -1, :]}
#                 pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
#                 # --- Loss Calculation ---
                
#                 # # Loss 1: Action Diffusion Loss
#                 # loss_diff = F.mse_loss(pred_noise, noise)
                
#                 # # Loss 2: Consistency Loss (Brain Completion)
#                 # # 使用原始 batch (包含未 Mask 的数据)，函数内部会自己处理 Student/Teacher 构建
#                 # loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                
#                 # # Loss 3: Distillation Regularization (Don't forget semantics)
#                 # # 强迫当前 Mask 状态下的 encoder_out 依然能恢复出全局语义
#                 # # 这完全复用了 Stage B 的逻辑
#                 # loss_distill_reg, _ = distill_fn(encoder_out, teacher_feats)


#                 # 1. 改为 reduction='none' 以便手动加权
#                 loss_diff_raw = F.mse_loss(pred_noise, noise, reduction='none') 
                
#                 # 2. 创建权重矩阵 (默认全是 1.0)
#                 # shape: [Batch, Pred_Horizon, Action_Dim] -> [B, 64, 8]
#                 loss_weights = torch.ones_like(loss_diff_raw)
                
#                 # 4. 计算加权后的均值
#                 loss_diff = (loss_diff_raw * loss_weights).mean()
#                 # =================================================================

#                 # Loss 2: Consistency Loss
#                 loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                
#                 # Loss 3: Distillation Regularization
#                 loss_distill_reg, _ = distill_fn(encoder_out, teacher_feats)
                
#                 # 🌟 组合 Loss
#                 # diff: 1.0 (主任务)
#                 # cons: 0.1 (辅助脑补)
#                 # distill: 0.05 (辅助语义锚定，防止漂移)
#                 total_loss = loss_diff + 0.1 * loss_cons + 0.05 * loss_distill_reg
                
#                 # 梯度累积归一化
#                 total_loss = total_loss / args.gradient_accumulation_steps

#             # --- 反向传播 ---
#             total_loss.backward()

#             # --- 参数更新 ---
#             if (i + 1) % args.gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
#                 optimizer.step()
#                 optimizer.zero_grad()
                
#                 global_step += 1
                
#                 # --- 日志记录 ---
#                 if global_step % 10 == 0:
#                     real_loss = total_loss.item() * args.gradient_accumulation_steps
                    
#                     print(f"Step [{global_step}/{args.max_train_steps}] Loss: {real_loss:.4f} | Diff: {loss_diff.item():.4f} | Cons: {loss_cons.item():.4f} | Reg: {loss_distill_reg.item():.4f}")
                    
#                     tb_writer.add_scalar('Train/Total_Loss', real_loss, global_step)
#                     if args.use_wandb and HAS_WANDB:
#                         wandb.log({
#                             "total_loss": real_loss,
#                             "diff_loss": loss_diff.item(),
#                             "cons_loss": loss_cons.item(),
#                             "distill_reg_loss": loss_distill_reg.item(),
#                             "global_step": global_step,
#                             "epoch": epoch,
#                             "lr": optimizer.param_groups[0]['lr']
#                         }, step=global_step)

#                 # --- 视频可视化 ---
#                 if global_step % 500 == 0 and args.use_wandb and HAS_WANDB:
#                     try:
#                         # 可视化真正喂给 RDT 的数据 (video_input)
#                         vid_sample = video_input[0].float().cpu().numpy() 
#                         main_view = np.transpose(vid_sample[0], (1, 0, 2, 3))
#                         wrist_view = np.transpose(vid_sample[1], (1, 0, 2, 3))
#                         combined_view = np.concatenate([main_view, wrist_view], axis=3) 
#                         wandb.log({
#                             "input_monitor": wandb.Video((combined_view * 255).astype(np.uint8), fps=4, format="gif", caption=f"S{global_step}: {mask_type}")
#                         }, step=global_step)
#                     except: pass

#                 # --- Checkpoint 保存 ---
#                 if global_step % args.checkpointing_steps == 0:
#                     save_path = os.path.join(args.output_dir, f"12stageC_step_{global_step}.pt")
#                     torch.save({
#                         'epoch': epoch,
#                         'global_step': global_step, 
#                         'rdt_state_dict': rdt_wrapper.state_dict(),
#                         'encoder_state_dict': fusion_encoder.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'pred_horizon': args.pred_horizon
#                     }, save_path)
#                     print(f"💾 Checkpoint saved: {save_path}")

#                 # --- 结束训练 ---
#                 if global_step >= args.max_train_steps:
#                     print(f"🎉 Reached target {args.max_train_steps} steps. Training Finished.")
#                     final_path = os.path.join(args.output_dir, f"stageC_final_{global_step}.pt")
#                     torch.save({
#                         'epoch': epoch,
#                         'global_step': global_step,
#                         'rdt_state_dict': rdt_wrapper.state_dict(),
#                         'encoder_state_dict': fusion_encoder.state_dict()
#                     }, final_path)
#                     tb_writer.close()
#                     if args.use_wandb and HAS_WANDB: wandb.finish()
#                     return 

#     tb_writer.close()
#     if args.use_wandb and HAS_WANDB: wandb.finish()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/data/12pick_up_the_orange_ball.hdf5')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/16checkpoints')
#     parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/checkpoints/16stageB_step_2000.pt')
    
#     # 物理 Batch Size (显存限制，保持 16)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--pred_horizon', type=int, default=64)
    
#     # === 关键控制参数 ===
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
#                         help="Number of updates steps to accumulate before update pass. (Effective BS = batch_size * this)")
    
#     parser.add_argument('--max_train_steps', type=int, default=10000, 
#                         help="Total number of training steps (parameter updates) to perform.")
    
#     parser.add_argument('--checkpointing_steps', type=int, default=500, 
#                         help="Save checkpoint every X updates.")
    
#     parser.add_argument('--resume_from_checkpoint', type=str, default=None)
#     parser.add_argument('--use_wandb', action='store_true', default=False)
    
#     args = parser.parse_args()
#     train_stage_c(args)

# train/stageC_joint.py
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
from losses.distillation_loss import DistillationLoss

# === 路径配置 ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
# 🟢 请确保这里指向正确的统计文件
STATS_PATH = '/yanghaochuan/data/121dataset_stats.json' 

def train_stage_c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 0. 初始化日志
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="RDT-StageC-Joint",
            name=f"ForeSight_StageC_{int(time.time())}",
            config=vars(args),
            resume="allow"
        )
    
    print(f"=== ForeSight VLA Training (Stage C: Policy Learning) ===")
    
    # 1. 模型加载
    print("Loading Models...")
    # 确保 teacher_dim 和 rdt_dim 与 Stage B 一致
    fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152, rdt_dim=768).to(device)
    
    # 加载 Stage B 预训练权重 (World Model)
    if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
        print(f"Loading Stage B (World Model): {args.stage_b_ckpt}")
        ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
        
        # 兼容只保存了 state_dict 或完整 checkpoint 的情况
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'encoder_state_dict' in ckpt:
            state_dict = ckpt['encoder_state_dict']
        else:
            state_dict = ckpt
            
        # 去除 module. 前缀 (如果是 DDP 训练保存的)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."): new_state_dict[k[7:]] = v
            else: new_state_dict[k] = v
            
        msg = fusion_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Stage B Loaded. Missing keys: {len(msg.missing_keys)}")
    else:
        print("⚠️ Warning: No Stage B checkpoint loaded! Training from scratch (Not Recommended).")
    
    # 冻结 VideoMAE Backbone，微调其他部分
    # 注意：这里我们让 Encoder 处于 eval 模式 (BN 不更新)，但参数 requires_grad=True (权重微调)
    fusion_encoder.eval() 
    for param in fusion_encoder.parameters(): param.requires_grad = True 
    for param in fusion_encoder.backbone.parameters(): param.requires_grad = False
    if fusion_encoder.text_encoder:
        for p in fusion_encoder.text_encoder.parameters(): p.requires_grad = False

    # 加载 RDT Policy
    rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, pred_horizon=args.pred_horizon).to(device)
    
    # RDT 权重加载
    if os.path.exists(RDT_PATH) or os.path.exists(os.path.join(RDT_PATH, "pytorch_model.bin")):
        rdt_file = RDT_PATH if os.path.isfile(RDT_PATH) else os.path.join(RDT_PATH, "pytorch_model.bin")
        if os.path.exists(rdt_file):
            print("Loading RDT pretrained weights...")
            state_dict = torch.load(rdt_file, map_location='cpu')
            rdt_wrapper.rdt_model.load_state_dict(state_dict, strict=False)

    # LoRA 配置
    print("Applying LoRA to RDT...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
        lora_dropout=0.05, bias="none"
    )
    rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    
    # 🟢 编译 Encoder (Backbone 冻结了，编译效果很好)
    print("🚀 Compiling FusionEncoder...")
    try:
        fusion_encoder = torch.compile(fusion_encoder)
    except Exception as e:
        print(f"⚠️ Encoder compilation failed: {e}")

    # # 🟢 编译 RDT (LoRA 部分可能需要一点时间编译)
    # print("🚀 Compiling RDT...")
    # try:
    #     rdt_wrapper.rdt_model = torch.compile(rdt_wrapper.rdt_model)
    # except Exception as e:
    #     print(f"⚠️ RDT compilation failed: {e}")


    # 优化器配置：RDT 学习率稍高，Encoder 学习率极低 (微调)
    params = [
        {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5}
    ]
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")
    distill_fn = DistillationLoss()

    # 3. 数据加载
    print(f"Loading Dataset from {args.data_root}")
    # 🟢 [关键修改] window_size 必须改为 6，与 Stage B 保持一致！
    dataset = RobotDataset(
        hdf5_path=args.data_root, 
        window_size=6,             # <--- Modified: Match Stage B
        pred_horizon=args.pred_horizon, 
        stats_path=STATS_PATH
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # 4. 续训逻辑
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

    # 5. 训练循环
    print(">>> Training Started <<<")
    
    # 无限 Epoch 循环，由 max_train_steps 终止
    total_epochs = 999999 
    
    for epoch in range(start_epoch, total_epochs):
        rdt_wrapper.train()
        
        for i, batch in enumerate(loader):
            if epoch == start_epoch and i < resume_batch_idx: continue

            # 数据搬运
            video = batch['video'].to(device, non_blocking=True) # [B, 3, 6, H, W]
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            actions = batch['action_target'].to(device, non_blocking=True)
            
            # 🟢 [ForeSight] 未来目标
            future_exo_target = batch['future_exo_target'].to(device, non_blocking=True)

            # Teacher Features (Distillation)
            real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
            real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            siglip_target = torch.mean(real_siglip, dim=1)
            exo_target = torch.mean(real_exo, dim=1)
            teacher_feats = {"siglip_features": siglip_target, "exo_features": exo_target}

            # Modality Dropout (随机 Mask 模拟推理时的不确定性)
            # rand_val = torch.rand(1).item()
            # video_input = video.clone()
            # ff_input = ff.clone()
            
            # if rand_val < 0.7: 
            #     video_input[:, 0] = 0.0 # Mask Main Camera
            #     ff_input[:, 0] = 0.0
            # elif rand_val < 0.8: 
            #     video_input[:, 1] = 0.0 # Mask Wrist Camera
            #     ff_input[:, 1] = 0.0
            rand_val = torch.rand(1).item()
            mask_type = "Wrist_Only" # 默认状态
            
            # video_input = video.clone()
            # ff_input = ff.clone()
            
            # # 策略：90% 的时间完全 Mask 掉 Main View
            # # 理由：推理时你只有 Wrist。如果训练时让它看到 Main，它就会依赖 Main。
            # # 必须把它逼到“只能靠 Wrist + Latent”来决策的绝境。
            # if rand_val < 1.01:
            #     video_input[:, 0] = 0.0
            #     ff_input[:, 0] = 0.0
            #     mask_type = "Simulate_Inference"
            
            # # 剩下 10%：Teacher Guidance (全可见)
            # # 仅用于维持 Encoder 的特征稳定性，不让它彻底遗忘 Stage B 学到的全图特征。
            # else:
            #     mask_type = "Teacher_Guidance"



            rand_val = torch.rand(1).item()
            
            video_input = video.clone()
            ff_input = ff.clone()
            
            if rand_val < 0.5:
                # [Mode A: Inference Simulation] (50%)
                # 模拟真实推理：Main Camera 丢失，只有 Wrist Camera
                # 目的：适应部分可观测环境
                video_input[:, 0] = 0.0 # Mask Main
                ff_input[:, 0] = 0.0    # Mask First Frame Main
                mask_type = "Inference_Mode (Wrist Only)"
                
            elif rand_val < 0.8:
                # [Mode B: Total Blindness] (30%)
                # 模拟全盲：Main + Wrist 全部丢失
                # 目的：强迫模型必须依赖 State (Proprioception)
                # 此时 Encoder 输出的 e_t 几乎没有视觉信息，Action 生成全靠 State Injection
                video_input[:] = 0.0 
                ff_input[:] = 0.0
                mask_type = "Blind_Mode (State Only)"
                
            else:
                # [Mode C: Teacher Guidance] (20%)
                # 全可见：Main + Wrist 都有
                # 目的：维持 VideoMAE 的特征提取能力，防止灾难性遗忘，并提供语义锚点
                mask_type = "Teacher_Mode (Full View)"


            CONSISTENCY_FREQ = 5


            with autocast('cuda', dtype=torch.bfloat16):
                # 1. Encoder Forward
                # 这里的 out 包含 'e_t' (70 tokens) 和 'wm_latents' (6 latents)
                encoder_out = fusion_encoder(video_input, text, state, ff_input)
                
                e_t = encoder_out['e_t']         # [B, 70, 768] -> 给 RDT
                wm_pred = encoder_out['wm_latents'] # [B, 6, 1152] -> 给 WM Loss
                
                # 2. RDT Forward (Action Generation)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                # Condition 传入 e_t 和 当前 state
                conditions = {"e_t": e_t, "state": state[:, -1, :]}
                pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
                # --- Loss Calculation ---
                
                # Loss 1: Action Diffusion Loss
                loss_diff = F.mse_loss(pred_noise, noise)
                # # 🟢 [修改] 稀疏计算 Consistency Loss
                # if global_step % CONSISTENCY_FREQ == 0:
                #     loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                # else:
                #     loss_cons = torch.tensor(0.0, device=device, requires_grad=True)
                # ✅ 修改后：直接禁用！
                # 我们不希望模型在“看不见”的时候去瞎猜“看得见”的特征，这会导致它产生幻觉。
                loss_cons = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Loss 2: 🟢 [ForeSight] World Model Loss (MSE + Cosine)
                # 必须与 Stage B 保持一致，防止微调时破坏 Latent 结构
                l_wm_mse = F.mse_loss(wm_pred, future_exo_target)
                
                wm_pred_norm = F.normalize(wm_pred, dim=-1)
                target_norm = F.normalize(future_exo_target, dim=-1)
                l_wm_cos = (1.0 - (wm_pred_norm * target_norm).sum(dim=-1)).mean()
                
                loss_wm = l_wm_mse + 0.5 * l_wm_cos
                
                # Loss 3: Regularization (Consistency & Distill)
                loss_cons = compute_consistency_loss(fusion_encoder, batch, device)
                loss_distill_reg, _ = distill_fn(encoder_out, teacher_feats)
                
                # 🌟 总 Loss
                # Diff: 1.0 (主任务)
                # WM: 0.5 (强约束，保持预测能力)
                # Cons: 0.1 (辅助)
                # Distill: 0.05 (防漂移)
                total_loss = loss_diff + 0.5 * loss_wm + 0.1 * loss_cons + 0.05 * loss_distill_reg
                total_loss = total_loss / args.gradient_accumulation_steps

            total_loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    real_loss = total_loss.item() * args.gradient_accumulation_steps
                    print(f"Step {global_step} | L: {real_loss:.4f} | Act: {loss_diff.item():.4f} | WM: {loss_wm.item():.4f} (Cos:{l_wm_cos.item():.3f})")
                    
                    tb_writer.add_scalar('Train/Total_Loss', real_loss, global_step)
                    if args.use_wandb and HAS_WANDB:
                        wandb.log({
                            "total_loss": real_loss,
                            "action_loss": loss_diff.item(),
                            "wm_loss": loss_wm.item(),
                            "wm_cos": l_wm_cos.item(),
                            "cons_loss": loss_cons.item(),
                            "global_step": global_step,
                            "epoch": epoch
                        }, step=global_step)

                # Checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"StageC_ForeSight_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step, 
                        'rdt_state_dict': rdt_wrapper.state_dict(),
                        'encoder_state_dict': fusion_encoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'pred_horizon': args.pred_horizon
                    }, save_path)
                    print(f"💾 Checkpoint saved: {save_path}")

                if global_step >= args.max_train_steps:
                    print(f"🎉 Training Finished.")
                    final_path = os.path.join(args.output_dir, f"StageC_ForeSight_final.pt")
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
    # 默认参数仅供参考，建议通过 shell 脚本传入
    parser.add_argument('--data_root', type=str, default='/yanghaochuan/data/hdf5/pick_up_the_orange_ball_and_put_it_on_the_plank.hdf5')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/121checkpoints_finetune')
    # 默认加载 Stage B (ForeSight Pretrained)
    parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/checkpoints/120StageB_ForeSight_step_2500.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pred_horizon', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--max_train_steps', type=int, default=10000)
    parser.add_argument('--checkpointing_steps', type=int, default=500)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()
    train_stage_c(args)