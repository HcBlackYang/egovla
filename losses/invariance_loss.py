# losses/invariance_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class InvarianceLoss(nn.Module):
    """
    背景不变性损失。
    当背景相同时，即使任务（前景）不同，背景上下文表征也应该相似。
    这需要特殊的数据增强（背景置换）。
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, background_context_1, background_context_2):
        """
        background_context_1: 原始样本的背景上下文 [B, 1, D]。
        background_context_2: 经过背景置换后的同一样本的背景上下文 [B, 1, D]。
                              理论上它应该与原始背景上下文保持一致。
        """
        return self.loss_fn(background_context_1, background_context_2)