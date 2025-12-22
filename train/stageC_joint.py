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

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from models.fusion_encoder import FusionEncoder
# from models.rdt_model import RDTWrapper
# from utils.dataset_loader import RobotDataset

# VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
# RDT_PATH = '/yanghaochuan/models/rdt-1b'

# def train_stage_c(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"=== Stage C Training: RDT Policy Learning on {device} ===")
#     print(f"=== Mode: BF16 | Workers: {args.num_workers} | Batch: {args.batch_size} ===")

#     # 1. 模型初始化
#     print("Loading FusionEncoder...")
#     fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
#     if os.path.exists(args.stage_b_ckpt):
#         print(f"Loading Stage B Weights: {args.stage_b_ckpt}")
#         fusion_encoder.load_state_dict(torch.load(args.stage_b_ckpt), strict=False)
#     else:
#         print(f"Warning: Stage B Checkpoint not found at {args.stage_b_ckpt}!")
#         print("Please ensure you have trained Stage B first.")
    
#     print("Loading RDT Policy...")
#     rdt_model = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768).to(device)

#     # 冻结 FusionEncoder
#     print("Freezing FusionEncoder...")
#     fusion_encoder.eval()
#     for param in fusion_encoder.parameters():
#         param.requires_grad = False
        
#     print("RDT is trainable.")
    
#     # 2. 调度器与优化器
#     noise_scheduler = DDPMScheduler(
#         num_train_timesteps=1000,
#         beta_schedule="squaredcos_cap_v2",
#         prediction_type="sample"
#     )
    
#     optimizer = optim.AdamW(rdt_model.parameters(), lr=1e-4, weight_decay=1e-6)

#     # 3. 数据加载
#     print(f"Loading data from: {args.data_root}")
#     if not os.path.exists(args.data_root):
#         raise FileNotFoundError(f"Data file not found at {args.data_root}")

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
#     rdt_model.train()
#     print(">>> Training Started... <<<")
    
#     for epoch in range(args.epochs):
#         total_loss = 0
#         start_time = time.time()
        
#         for i, batch in enumerate(loader):
#             # 准备数据
#             video = batch['video'].to(device, non_blocking=True)
#             state = batch['state'].to(device, non_blocking=True) # 已归一化的 [B, 16, 7/8]
#             text = batch['text_tokens'].to(device, non_blocking=True)
#             ff = batch['first_frame'].to(device, non_blocking=True)
            
#             # Target Actions (Next Step)
#             actions = batch['action_target'].to(device, non_blocking=True)

#             optimizer.zero_grad()

#             with autocast('cuda', dtype=torch.bfloat16):
#                 # 提取特征
#                 with torch.no_grad():
#                     encoder_outputs = fusion_encoder(video, text, state, ff)
#                     e_t = encoder_outputs['e_t'] # [B, 768]
                
#                 # 加噪
#                 timesteps = torch.randint(
#                     0, noise_scheduler.config.num_train_timesteps, 
#                     (actions.shape[0],), device=device
#                 ).long()
                
#                 noise = torch.randn_like(actions)
#                 noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
#                 # 预测
#                 conditions = {"e_t": e_t}
#                 pred_noise = rdt_model(noisy_actions, timesteps, conditions)
                
#                 if pred_noise.dim() == 3 and noise.dim() == 2:
#                     noise = noise.unsqueeze(1) 
                    
#                 loss = torch.nn.functional.mse_loss(pred_noise, noise)

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(rdt_model.parameters(), 1.0)
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             if i % 10 == 0 and i > 0:
#                 elapsed = time.time() - start_time
#                 speed = (i * args.batch_size) / elapsed
#                 print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f} | Speed: {speed:.1f} samples/s")

#         # 每个 Epoch 保存一次
#         os.makedirs(args.output_dir, exist_ok=True)
#         ckpt_name = f"stageC_epoch_{epoch}.pt"
#         save_path = os.path.join(args.output_dir, ckpt_name)
#         torch.save({
#             'rdt_state_dict': rdt_model.state_dict(),
#             'encoder_state_dict': fusion_encoder.state_dict() # 保存 Encoder 以备推理时使用
#         }, save_path)
#         print(f"Saved Checkpoint to {save_path}")

#     # 保存最终模型
#     final_path = os.path.join(args.output_dir, "stageC_final.pt")
#     torch.save({
#         'rdt_state_dict': rdt_model.state_dict(),
#         'encoder_state_dict': fusion_encoder.state_dict()
#     }, final_path)
#     print(f"Training Finished. Saved final model to {final_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
    
#     # === 修改部分：设置了默认参数并去掉了 required=True ===
#     parser.add_argument('--data_root', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
#     parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/projects/checkpoints/stageB_papercup.pt')
#     parser.add_argument('--output_dir', type=str, default='/yanghaochuan/projects/checkpoints')
#     parser.add_argument('--batch_size', type=int, default=48)
#     parser.add_argument('--num_workers', type=int, default=16)
#     parser.add_argument('--epochs', type=int, default=5) # 默认 5 个 epoch
    
#     args = parser.parse_args()
#     train_stage_c(args)

# train/stageC_joint.py
import sys
import os
import torch
import torch.optim as optim
import argparse
import time
from torch.utils.data import DataLoader
from torch.amp import autocast
from diffusers import DDPMScheduler
# === 新增：导入 PEFT ===
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_encoder import FusionEncoder
from models.rdt_model import RDTWrapper
from utils.dataset_loader import RobotDataset

VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'
RDT_PATH = '/yanghaochuan/models/rdt-1b'

def train_stage_c(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage C Training: RDT Policy (LoRA) on {device} ===")
    
    # 1. 模型初始化
    print("Loading FusionEncoder...")
    fusion_encoder = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
    if os.path.exists(args.stage_b_ckpt):
        print(f"Loading Stage B Weights: {args.stage_b_ckpt}")
        fusion_encoder.load_state_dict(torch.load(args.stage_b_ckpt), strict=False)
    else:
        print(f"Warning: Stage B Checkpoint not found at {args.stage_b_ckpt}!")
    
    print("Freezing FusionEncoder...")
    fusion_encoder.eval()
    for param in fusion_encoder.parameters():
        param.requires_grad = False
    
    print("Loading RDT Policy...")
    rdt_wrapper = RDTWrapper(action_dim=8, model_path=RDT_PATH, rdt_cond_dim=768).to(device)
    
    # === 核心修改：应用 LoRA ===
    print("Applying LoRA to RDT...")
    # 针对 Transformer 的常见 Linear 层名称
    # 如果报错找不到层，可以打印 rdt_wrapper.rdt_model 看看具体层名
    peft_config = LoraConfig(
        r=16,               # Rank
        lora_alpha=32,      # Alpha
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "linear"], 
        lora_dropout=0.05,
        bias="none",
        # task_type="FEATURE_EXTRACTION" # RDT 本质是去噪预测
    )
    
    # 我们只对 RDT 内部的核心模型加 LoRA，而不是 Wrapper 壳子
    # 这里的 rdt_wrapper.rdt_model 是实际的 Transformer
    try:
        rdt_wrapper.rdt_model = get_peft_model(rdt_wrapper.rdt_model, peft_config)
        rdt_wrapper.rdt_model.print_trainable_parameters()
    except Exception as e:
        print(f"LoRA 自动挂载失败 (可能是层名不匹配): {e}")
        print("尝试全量微调 (不推荐)...")
        # 如果自动挂载失败，就fallback到普通训练
    
    # 2. 调度器与优化器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample"
    )
    
    # 只优化 LoRA 参数 (requires_grad=True 的参数)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, rdt_wrapper.parameters()), lr=1e-4, weight_decay=1e-4)

    # 3. 数据加载 (会自动读取 stats.json)
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

    # 4. 训练循环
    rdt_wrapper.train()
    print(">>> LoRA Training Started... <<<")
    
    for epoch in range(args.epochs):
        total_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            video = batch['video'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True) 
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            actions = batch['action_target'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    encoder_outputs = fusion_encoder(video, text, state, ff)
                    e_t = encoder_outputs['e_t'] 
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=device).long()
                noise = torch.randn_like(actions)
                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
                
                conditions = {"e_t": e_t}
                pred_noise = rdt_wrapper(noisy_actions, timesteps, conditions)
                
                if pred_noise.dim() == 3 and noise.dim() == 2:
                    noise = noise.unsqueeze(1) 
                    
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(rdt_wrapper.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                speed = (i * args.batch_size) / elapsed
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f} | Speed: {speed:.1f} samples/s")

        # 保存
        os.makedirs(args.output_dir, exist_ok=True)
        # 注意：使用 LoRA 时，state_dict 会只包含 LoRA 权重，文件很小
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f"stageC_lora_epoch_{epoch}.pt")
            torch.save({
                'rdt_state_dict': rdt_wrapper.state_dict(), # 包含 LoRA 权重
                'encoder_state_dict': fusion_encoder.state_dict()
            }, save_path)
            print(f"Saved Checkpoint to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/yanghaochuan/projects/data/pick_up_the_paper_cup.hdf5')
    parser.add_argument('--stage_b_ckpt', type=str, default='/yanghaochuan/projects/checkpoints/stageB_papercup.pt')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/projects/checkpoints')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50) # LoRA 建议多训几轮
    
    args = parser.parse_args()
    train_stage_c(args)