# losses/temporal_consistency.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    """
    跨帧一致性与轨迹对比损失。
    鼓励模型生成的表征在时间上是平滑的。
    """
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, temporal_features):
        """
        temporal_features: 模型的时序输出，如 temporal_head_output。
                           期望形状为 [B, T, D]。
        """
        # 1. 维度检查与兼容性处理
        # 如果输入是 2D [B, D]，说明只有全局特征，无法计算时序差分
        if temporal_features.dim() == 2:
            return torch.tensor(0.0, device=temporal_features.device, requires_grad=True)

        # 如果时间维度 (Dim 1) 长度 <= 1，也无法计算差分
        if temporal_features.size(1) <= 1:
            return torch.tensor(0.0, device=temporal_features.device, requires_grad=True)

        # 2. 计算连续帧之间的差异
        diff = temporal_features[:, 1:, :] - temporal_features[:, :-1, :]
        
        if self.loss_type == 'l2':
            loss = torch.mean(diff**2)
        elif self.loss_type == 'l1':
            loss = torch.mean(torch.abs(diff))
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
            
        return loss