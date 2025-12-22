# losses/distillation_loss.py

import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    双教师蒸馏损失。
    - 语义头对齐 SigLIP 教师。
    - 时序头对齐 Exo 教师。
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        # alpha 控制语义损失和时序损失的权重
        self.alpha = alpha
        self.semantic_loss_fn = nn.MSELoss()
        self.temporal_loss_fn = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs):
        """
        student_outputs: fusion_encoder的输出字典。
        teacher_outputs: 包含教师模型特征的字典。
        """
        semantic_student = student_outputs["semantic_head_output"]
        temporal_student = student_outputs["temporal_head_output"]
        
        siglip_teacher = teacher_outputs["siglip_features"] # [B, T, D]
        exo_teacher = teacher_outputs["exo_features"]       # [B, T, D]

        # 计算语义损失 (与SigLIP对齐)
        loss_semantic = self.semantic_loss_fn(semantic_student, siglip_teacher)
        
        # 计算时序损失 (与Exo教师对齐)
        loss_temporal = self.temporal_loss_fn(temporal_student, exo_teacher)
        
        # 组合损失
        total_loss = self.alpha * loss_semantic + (1 - self.alpha) * loss_temporal
        
        return total_loss, {"loss_semantic": loss_semantic, "loss_temporal": loss_temporal}