# import torch
# import torch.nn.functional as F

# def compute_consistency_loss(fusion_encoder, batch, device):
#     """
#     计算单双摄特征一致性损失 (Invariance/Consistency Loss)。
    
#     原理:
#         1. Input A (Dual): 完整的 [Main, Wrist] 视频流。
#         2. Input B (Single): 将 Main 视角全黑 masked 掉的 [Black, Wrist] 视频流。
#         3. 目标: 让 Input B 的特征 (Student) 逼近 Input A 的特征 (Teacher/Target)。
        
#     Args:
#         fusion_encoder: 您的特征提取器模型 (FusionEncoder)
#         batch: DataLoader 出来的一个 batch 字典
#         device: torch device
        
#     Returns:
#         loss: 标量 Tensor (MSE)
#     """
    
#     # 1. 获取原始数据 (Expect [B, 2, 3, 16, 224, 224])
#     # 注意：为了使用此 Loss，您的 Dataset 必须返回双摄数据。
#     # 如果 Dataset 返回的是 [B, 3, 16, 224, 224] (单摄)，则需要先修改 Dataset Loader。
#     video = batch['video'].to(device, non_blocking=True)
#     text = batch['text_tokens'].to(device, non_blocking=True)
#     state = batch['state'].to(device, non_blocking=True)
#     ff = batch['first_frame'].to(device, non_blocking=True) 

#     # 检查维度: 必须包含 View 维度 (通常是第1维)
#     # [Batch, View, Channel, Time, Height, Width]
#     if video.dim() == 6 and video.shape[1] == 2:
#         # === A. 构造双摄输入 (Dual View - Teacher) ===
#         # 老师看到所有信息
#         video_dual = video 
        
#         # === B. 构造单摄输入 (Single View - Student) ===
#         # 学生的主摄被遮挡 (模拟推理情况)
#         video_single = video.clone()
#         video_single[:, 0] = 0.0 # 将 View 0 (Main Camera) 设为全黑
        
#         # === C. 前向传播 ===
        
#         # 1. 提取 Teacher 特征 (不需要梯度)
#         with torch.no_grad():
#             # 注意：fusion_encoder 内部需要处理 [B, 2, ...] 的输入
#             # 如果 encoder 只接受 [B, C, ...], 它内部应该有 flatten B*V 的逻辑
#             # 或者我们在外面 flatten 也可以，但通常建议放在 Encoder forward 里统一处理
#             out_dual = fusion_encoder(video_dual, text, state, ff)
#             feat_dual = out_dual['e_t'].detach() # [B, 64, 768] (Stop Gradient)
            
#         # 2. 提取 Student 特征 (需要梯度)
#         out_single = fusion_encoder(video_single, text, state, ff)
#         feat_single = out_single['e_t'] # [B, 64, 768]
        
#         # === D. 计算损失 ===
#         # 强制 Student 模仿 Teacher
#         loss = F.mse_loss(feat_single, feat_dual)
        
#         return loss

#     else:
#         # 如果数据不是双摄格式，无法计算此 Loss
#         # 这种情况下返回 0，避免报错，但请务必检查 Dataset
#         # print("[Warning] Consistency Loss skipped: Input is not dual-view.")
#         return torch.tensor(0.0, device=device, requires_grad=True)

import torch
import torch.nn.functional as F

def compute_consistency_loss(fusion_encoder, batch, device):
    """
    计算单双摄特征一致性损失 (Invariance/Consistency Loss)。
    
    原理:
        1. Input A (Dual): 完整的 [Main, Wrist] 视频流 + 完整的首帧。
        2. Input B (Single): [Black, Wrist] 视频流 + [Black, Wrist] 首帧。
        3. 目标: 让 Input B 的特征 (Student) 逼近 Input A 的特征 (Teacher/Target)。
    """
    
    # 1. 获取原始数据 (Expect [B, 2, 3, 16, 224, 224])
    video = batch['video'].to(device, non_blocking=True)
    text = batch['text_tokens'].to(device, non_blocking=True)
    state = batch['state'].to(device, non_blocking=True)
    ff = batch['first_frame'].to(device, non_blocking=True) 

    # 检查维度: 必须包含 View 维度
    if video.dim() == 6 and video.shape[1] == 2:
        # === A. 构造双摄输入 (Dual View - Teacher) ===
        video_dual = video 
        ff_dual = ff
        
        # === B. 构造单摄输入 (Single View - Student) ===
        # 学生的主摄被遮挡 (模拟推理情况)
        video_single = video.clone()
        video_single[:, 0] = 0.0 # 将 View 0 (Main Camera) 视频流设为全黑
        
        # 🚨 [关键修改] 同步 Mask 首帧！
        # 如果不Mask这里，学生会“作弊”，通过看高清首帧来推测全局信息，
        # 导致它学不会从手腕视频流中推断位置。
        ff_single = ff.clone()
        ff_single[:, 0] = 0.0 # 将 View 0 (Main Camera) 首帧设为全黑
        
        # === C. 前向传播 ===
        
        # 1. 提取 Teacher 特征 (不需要梯度)
        with torch.no_grad():
            out_dual = fusion_encoder(video_dual, text, state, ff_dual)
            feat_dual = out_dual['e_t'].detach() # [B, 64, 768] (Stop Gradient)
            
        # 2. 提取 Student 特征 (需要梯度)
        # 🚨 这里必须传入 masked 后的 ff_single
        out_single = fusion_encoder(video_single, text, state, ff_single)
        feat_single = out_single['e_t'] # [B, 64, 768]
        
        # === D. 计算损失 ===
        # 强制 Student 模仿 Teacher
        loss = F.mse_loss(feat_single, feat_dual)
        
        return loss

    else:
        return torch.tensor(0.0, device=device, requires_grad=True)