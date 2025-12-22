# losses/decoupling_regularizer.py
import torch
import torch.nn as nn

class DecouplingLoss(nn.Module):
    """
    任务/背景解耦损失。
    目标是让任务槽的信息和背景上下文的信息尽可能独立。
    同时，利用置信度来加权。
    """
    def __init__(self):
        super().__init__()
        # 使用简单的正则化项：最小化任务槽和背景上下文的点积
        # 也可以使用更复杂的互信息最小化方法

    def forward(self, task_slots, background_context, confidence):
        """
        task_slots: [B, N_slots, D]
        background_context: [B, 1, D]
        confidence: [B, N_slots, 1]
        """
        B, N, D = task_slots.shape
        
        # 将背景上下文广播到与任务槽相同的形状
        background_context = background_context.expand_as(task_slots)
        
        # 计算点积相似度
        # cos_sim = F.cosine_similarity(task_slots, background_context, dim=-1) # [B, N_slots]
        # 此处使用更简单的点积作为正则项
        dot_product = torch.sum(task_slots * background_context, dim=-1) # [B, N_slots]

        # 用置信度加权，我们希望高置信度的任务槽与背景更不相关
        weighted_dot_product = dot_product * confidence.squeeze(-1)
        
        # 损失是让这个加权点积的绝对值尽可能小
        loss = torch.mean(torch.abs(weighted_dot_product))
        
        return loss