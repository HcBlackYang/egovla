# train/stageB_train_fast.py
import sys
import os
import torch
import torch.optim as optim
import argparse
import time
from torch.utils.data import DataLoader
from torch.amp import autocast # BF16 不需要 GradScaler

# 强制使用 Flash Attention，禁用普通数学注意力
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False) 
torch.backends.cuda.enable_mem_efficient_sdp(True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_encoder import FusionEncoder
from losses.distillation_loss import DistillationLoss
from losses.decoupling_regularizer import DecouplingLoss
from losses.temporal_consistency import TemporalConsistencyLoss
from utils.dataset_loader import RobotDataset

VIDEO_MAE_PATH = '/yanghaochuan/models/VideoMAEv2-Large'

def train_stage_b(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage B Training: Real Teacher Distillation on {device} ===")
    print(f"=== Mode: BF16 (Fast & Stable) | Workers: {args.num_workers} | Batch: {args.batch_size} ===")
    
    # 启用 TF32 (在 A800 上能加速 FP32/BF16 计算)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. 初始化模型
    model = FusionEncoder(backbone_path=VIDEO_MAE_PATH, teacher_dim=1152).to(device)
    
    if os.path.exists(args.stage_a_ckpt):
        print(f"Loading Stage A: {args.stage_a_ckpt}")
        model.load_state_dict(torch.load(args.stage_a_ckpt), strict=False)

    # 冻结参数
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
    
    # === 优化点 A: 编译模型 (PyTorch 2.0+) ===
    print("Compiling model with torch.compile... (First step will be slow)")
    # try:
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(f"Compile failed, falling back to eager mode: {e}")

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
    
    # 学习率可以适当回升，因为 BF16 很稳，且 Batch Size 变大了
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    loss_weights = {"distill":1.0, "decouple":0.5, "consistency":1.0}

    # 4. 训练循环
    model.train()
    print(">>> Training Started... (BF16 Mode) <<<")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        for i, batch in enumerate(loader):
            video = batch['video'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True)
            text = batch['text_tokens'].to(device, non_blocking=True)
            ff = batch['first_frame'].to(device, non_blocking=True)
            
            real_siglip = batch['teacher_siglip'].to(device, non_blocking=True)
            real_exo = batch['teacher_exo'].to(device, non_blocking=True)
            
            # 时序平均
            siglip_target = torch.mean(real_siglip, dim=1)
            exo_target = torch.mean(real_exo, dim=1)
            
            # === 新增：给 Teacher 特征加噪声 (仅在训练时) ===
            if model.training:
                noise_scale = 0.01 # 视特征数值范围而定，通常 0.01-0.05
                siglip_target += torch.randn_like(siglip_target) * noise_scale
                exo_target += torch.randn_like(exo_target) * noise_scale
            # ============================================

            teacher_feats = {
                "siglip_features": siglip_target,
                "exo_features": exo_target
            }

            optimizer.zero_grad()

            # === 优化点 B: 启用 BFloat16 ===
            # A800 专属：不需要 Scaler，因为范围够大，不会溢出
            with autocast('cuda', dtype=torch.bfloat16):
                out = model(video, text, state, ff)
                
                l_distill, _ = distill_fn(out, teacher_feats)
                l_decouple = decouple_fn(out['task_slots'], out['background_context'], out['task_confidence'])
                l_time = temporal_fn(out['temporal_head_output'])
                
                loss = loss_weights['distill'] * l_distill + \
                       loss_weights['decouple'] * l_decouple + \
                       loss_weights['consistency'] * l_time

            # === 普通反向传播 (BF16 不需要 scaler.step) ===
            loss.backward()
            
            # 依然保留梯度裁剪以防万一
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 进度打印
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                speed = (i * args.batch_size) / elapsed
                print(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.8e} | Distill: {l_distill.item():.8e} | Speed: {speed:.1f} img/s")

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "1223stageB_papercup.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True) 
    parser.add_argument('--stage_a_ckpt', type=str, default='/yanghaochuan/checkpoints/stageA_final.pt')
    parser.add_argument('--output_dir', type=str, default='/yanghaochuan/checkpoints')
    # === 优化点 C: Batch Size 翻倍 ===
    parser.add_argument('--batch_size', type=int, default=24) # 直接上 48！
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    
    train_stage_b(args)