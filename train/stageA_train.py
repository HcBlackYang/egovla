# train/stageA_train.py

import torch
import torch.optim as optim
import argparse
import yaml
from torch.utils.data import DataLoader

# --- 使用占位符模拟导入 ---
from models.fusion_encoder import FusionEncoder # 复用上一阶段代码
from utils.dataset_loader import RobotDataset # 复用上一阶段代码
from losses.distillation_loss import DistillationLoss # 复用上一阶段代码
# 模拟一个简单的重建损失
ReconstructionLoss = torch.nn.MSELoss
# --- 占位符结束 ---

def train_stage_a(config, args):
    """阶段 A 的主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 初始化模型
    model = FusionEncoder(
        rdt_dim=config['model']['rdt_dim'],
        num_task_slots=config['model']['num_task_slots']
    ).to(device)

    # 2. 冻结主干网络
    if args.freeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        print("VideoMAE backbone has been frozen.")

    # 3. 准备数据
    # train_dataset = RobotDataset(data_root=config['data']['train_path'], window_size=config['data']['window_size'])
    # train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    # 此处使用伪数据加载器
    train_loader = [(torch.randn(config['training']['batch_size'], 16, 3, 224, 224),
                     torch.randn(config['training']['batch_size'], 1, 768),
                     torch.randn(config['training']['batch_size'], 16, 7),
                     torch.randn(config['training']['batch_size'], 10, 768)) for _ in range(10)] # 模拟10个batch
    
    print("数据加载器准备完毕。")
    
    # 4. 初始化损失函数和优化器
    loss_weights = yaml.safe_load(args.loss_weights)
    recon_loss_fn = ReconstructionLoss()
    semantic_loss_fn = DistillationLoss(alpha=1.0) # 此阶段只关心语义部分
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    
    print(f"开始阶段 A 训练，共 {args.epochs} 个 epochs...")
    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # 模拟数据解包和教师信号
            video, first_frame_summary, state, text_tokens = [d.to(device) for d in batch]
            teacher_outputs = {"siglip_features": torch.randn_like(video[:,:,0,0,:768])} # 模拟SigLIP特征
            
            optimizer.zero_grad()
            
            # 前向传播
            student_outputs = model(video, text_tokens, state, first_frame_summary)
            
            # 计算损失
            # 模拟重建损失：让e_t去重建第一帧摘要
            loss_recon = recon_loss_fn(student_outputs['e_t'], first_frame_summary.squeeze(1))
            
            # 计算语义蒸馏损失
            loss_semantic, _ = semantic_loss_fn(student_outputs, teacher_outputs)
            
            # 加权总损失
            combined_loss = loss_weights['recon'] * loss_recon + \
                            loss_weights['semantic'] * loss_semantic
            
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    print("阶段 A 训练完成。")
    # 保存模型权重
    # torch.save(model.state_dict(), "stageA_final.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="阶段 A 训练脚本")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--freeze_backbone', type=bool, default=True, help='是否冻结主干网络')
    parser.add_argument('--loss_weights', type=str, default='{"recon":1.0, "semantic":0.8}', help='损失权重的JSON字符串')
    
    args = parser.parse_args()
    
    # 创建一个虚拟的配置文件用于演示
    dummy_config = {
        'model': {'rdt_dim': 768, 'num_task_slots': 8},
        'training': {'batch_size': 4, 'learning_rate': 1e-4}
    }
    # with open(args.config, 'r') as f:
    #     config = yaml.safe_load(f)
    
    train_stage_a(dummy_config, args)