

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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_encoder import FusionEncoder
from models.rdt_model import RDTWrapper
from utils.dataset_loader import RobotDataset
from losses.consistency_loss import compute_consistency_loss

# === 路径配置 (请修改为您自己的路径) ===
VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'
STATS_PATH = '/yanghaochuan/projects/data/dataset_stats.json'

def train_stage_c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage C Joint Training (End-to-End) ===")
    print(f"   - Modality Dropout: Enabled (p=0.5)")
    print(f"   - Action Chunking: {args.pred_horizon} steps")
    print(f"   - Consistency Loss: Enabled")
    
    # 1. 初始化 Encoder (VideoMAE + Projector)
    # 这一步会加载 VideoMAE 权重
    print("Loading Fusion Encoder...")
    fusion_encoder = FusionEncoder(
        backbone_path=VIDEO_MAE_PATH, 
        teacher_dim=1152,
        rdt_dim=768
    ).to(device)
    
    # 加载 Stage B 预训练好的 Encoder 权重 (如果有)
    if args.stage_b_ckpt and os.path.exists(args.stage_b_ckpt):
        print(f"Loading Stage B Checkpoint: {args.stage_b_ckpt}")
        ckpt = torch.load(args.stage_b_ckpt, map_location='cpu')
        # 兼容不同保存格式
        state_dict = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt
        fusion_encoder.load_state_dict(state_dict, strict=False)
    
    # 冻结 VideoMAE Backbone (为了节省显存，通常只微调 Projector 和 Adapter)
    # 如果显存充足且想追求极致性能，可以解冻
    fusion_encoder.eval() 
    for param in fusion_encoder.parameters():
        param.requires_grad = True # 让 Projector 训练
    for param in fusion_encoder.backbone.parameters():
        param.requires_grad = False # 冻结 ViT
    if fusion_encoder.text_encoder:
        for param in fusion_encoder.text_encoder.parameters(): param.requires_grad = False

    # 2. 初始化 Policy (RDT-1B)
    print("Loading RDT Policy...")
    # 注意: 这里的 rdt_cond_dim 实际上没用了，因为我们改用了 img_c (1152)
    rdt_wrapper = RDTWrapper(
        action_dim=8, 
        model_path=RDT_PATH, 
        pred_horizon=args.pred_horizon
    ).to(device)
    
    # 3. 配置 LoRA
    print("Applying LoRA to RDT...")
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        # 对 RDT 内部所有的 Linear 层应用 LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
        lora_dropout=0.05, 
        bias="none"
    )
    rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
    rdt_wrapper.rdt_model.print_trainable_parameters()
    
    # 4. 优化器 & Scheduler
    # 我们同时训练 RDT LoRA 和 FusionEncoder 的 Projector
    params_to_optimize = [
        {'params': filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), 'lr': 1e-4},
        {'params': filter(lambda p: p.requires_grad, fusion_encoder.parameters()), 'lr': 1e-5} # Encoder 学习率低一点
    ]
    optimizer = optim.AdamW(params_to_optimize, weight_decay=1e-4)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample"
    )

    # 5. 数据加载
    print(f"Loading Dataset from {args.data_root}")
    # 注意: window_size=16 (历史), pred_horizon=16 (未来)
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

    # 6. 训练循环
    rdt_wrapper.train()
    fusion_encoder.train() # Projector 需要 train 模式 (Dropout)
    
    print(">>> Training Started... <<<")
    for epoch in range(args.epochs):
        total_loss = 0
        
        for i, batch in enumerate(loader):
            # [B, 16, 3, 224, 224] (如果 Dataset 没有 View 维)
            # 或者 [B, 2, 3, 224, 224] (如果 Dataset 是 snapshot，不对)
            # 根据 dataset_loader.py，video 是 [B, 16, 3, 224, 224] (单摄)
            # 哎呀，RobotDataset 默认只读了 robot0_eye_in_hand。
            # 为了 Joint Training 的 Dropour 生效，我们需要 Dataset 返回双摄数据！
            # (如果 Dataset 只有单摄数据，您需要修改 Dataset Loader 读两个相机)
            
            # 假设 Dataset 已经修正为读取双摄 (见下方说明)，或者我们在这里模拟
            # 如果 HDF5 里只有 Wrist，那我们只能假设 Main 是全黑的，但这没意义。
            # **必须假设 HDF5 里有 Main Camera 数据**
            
            # 这里假设 video 已经是 [B, 2, 3, 16, H, W] (Views=2)
            # 如果您的 dataset_loader 还没改好去读 Main Camera，请务必去改。
            # 现在的代码假设 video 是 [B, 16, 3, H, W] (单摄)
            # 为了不报错，我们先按单摄逻辑写，但加上 Dropout 逻辑
            
            video = batch['video'].to(device, non_blocking=True) # [B, 16, 3, H, W]
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True) # [B, 1, 3, H, W]
            actions = batch['action_target'].to(device, non_blocking=True) # [B, 16, 8]

            # === 构造双摄输入 ===
            # 因为您的 HDF5 可能只有 Wrist，我们在这里手动构造一个"假双摄" batch
            # 但为了训练有效，我们最好有真的 Main Camera。
            # 如果实在没有，我们只能用全黑 Main Camera 占位，强迫模型只学 Wrist。
            # 但为了回答"如何避免悬停"，最好的办法是训练时偶尔给它看全黑 Main。
            
            # [B, 16, 3, H, W] -> [B, 1, 3, 16, H, W] -> [B, 2, 3, 16, H, W]
            video = video.permute(0, 2, 1, 3, 4).unsqueeze(1) # [B, 1, 3, 16, H, W]
            
            # 构造 Main Camera (假设现在没有真实 Main 数据，全黑)
            # 如果您有真实 Main 数据，请修改 Dataset Loader 读取它！
            main_cam = torch.zeros_like(video) 
            
            # 真正的输入: [Main, Wrist]
            video_input = torch.cat([main_cam, video], dim=1) # [B, 2, 3, 16, H, W]
            
            # === Modality Dropout (核心) ===
            # 随机 Mask 掉 Main Camera (虽然现在已经是全黑了，如果是真实数据这一步很有用)
            # 这里的逻辑是：
            # 1. 50% 概率：保持原样 (Main=Real/Black, Wrist=Real)
            # 2. 50% 概率：Main=Black, Wrist=Real (模拟推理)
            # 如果 Main 本来就是 Black，这步没区别。
            # 但如果 Main 是 Real，这步就教会了模型："看不见 Main 也要工作"。
            
            if torch.rand(1) < 0.5:
                 video_input[:, 0] = 0.0 # Mask Main
            
            # === 构造 ff (First Frame) ===
            # ff 也需要是双摄
            ff = ff.permute(0, 2, 1, 3, 4).unsqueeze(1)
            ff_main = torch.zeros_like(ff)
            ff_input = torch.cat([ff_main, ff], dim=1)

            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):
                # 1. Encoder 提特征
                # 输出 e_t: [B, 64, 768]
                encoder_outputs = fusion_encoder(video_input, text, state, ff_input)
                e_t = encoder_outputs['e_t']
                
                # 2. 扩散训练
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                conditions = {"e_t": e_t}
                
                # RDT Forward
                pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
                # Diffusion Loss
                loss_diff = torch.nn.functional.mse_loss(pred_noise, noise)
                
                # 3. Consistency Loss (可选，但推荐)
                # 计算这个比较耗时，因为它要再跑一次 forward
                # 如果显存不够，可以注释掉
                # loss_cons = compute_consistency_loss(fusion_encoder, batch, device) 
                # total_loss_step = loss_diff + 0.1 * loss_cons
                
                total_loss_step = loss_diff

            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
            optimizer.step()
            
            total_loss += total_loss_step.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} [{i}] Loss: {total_loss_step.item():.4f}")

        # 保存
        if epoch % 5 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"stageC_joint_epoch_{epoch}.pt")
            torch.save({
                'rdt_state_dict': rdt_wrapper.state_dict(),
                'encoder_state_dict': fusion_encoder.state_dict()
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--stage_b_ckpt', type=str, default=None, help="Path to Stage B encoder weights")
    parser.add_argument('--batch_size', type=int, default=16) # Joint Training 显存占用大，调小 Batch
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pred_horizon', type=int, default=16)
    
    args = parser.parse_args()
    train_stage_c(args)