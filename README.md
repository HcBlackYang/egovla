# Ego-Exo Distilled RDT: Decoupled Diffusion Policy for VLA

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.29+-yellow.svg)](https://huggingface.co/docs/diffusers/index)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue.svg)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **核心思想**：通过 **Modality Dropout**（模态丢弃）与 **Action Chunking**（动作分块），实现基于“纯手腕视角”（Wrist-View Only）的高鲁棒性、高流畅度机器人操控模型。

本项目是一个端到端的具身智能（Embodied AI）系统，结合了 **VideoMAEv2** 的时序感知能力与 **Robotics Diffusion Transformer (RDT)** 的长序列动作生成能力。

---

## 🌟 主要特性 (Key Features)

### 👁️ 鲁棒的单摄推理 (Robust Single-View Inference)
- **Modality Dropout**: 训练时随机丢弃（Mask）主摄图像，强迫模型学习仅依赖手腕相机（Wrist Camera）进行决策。
- **Fake Main View**: 推理时构造全黑的主摄输入，与训练时的 Dropout 分布保持一致，彻底解决“悬停”和分布偏移问题。
- **Consistency Loss**: 引入一致性损失，强制单摄特征逼近双摄特征。

### 🌊 流畅的动作控制 (Smooth Action Chunking)
- **Sequence Prediction**: 模型一次预测未来 **16步 (Horizon=16)** 的动作序列，而非单步预测。
- **Async Execution**: 机器人异步执行动作序列，消除通信延迟导致的“卡顿” (Stop-and-Go)，实现丝滑操作。

### 🎓 双教师蒸馏架构 (Dual-Teacher Distillation)
- **语义教师**: 使用冻结的 **SigLIP** 提供开放世界语义理解。
- **时序教师**: 使用 **Exo-View** 特征强化动作细节捕捉。

### ⚡️ 高效联合训练 (Joint Training with LoRA)
- **End-to-End LoRA**: 冻结 VideoMAE Backbone，仅微调 Projector 和 RDT 的 LoRA 适配器。
- **Memory Efficient**: 支持在有限显存下进行端到端的多模态联合训练。

---

## 🏗️ 系统架构

系统分为三个主要阶段：

1. **Stage B (Feature Alignment)**: 预训练 `FusionEncoder`
   - 训练 Projector 和解耦路由层，使其特征逼近教师模型 (SigLIP/Exo)。

2. **Compute Stats (Normalization)**: 数据统计
   - 计算动作空间的均值和标准差，采用 **Z-Score** 归一化，确保动作输出的精准度。

3. **Stage C (Joint Training)**: 联合训练 `RDT Policy`
   - **输入**: 双摄视频 (Main + Wrist)。
   - **机制**: 实时进行 Modality Dropout (随机抹黑 Main)。
   - **输出**: 16步动作序列 (Action Chunk)。
   - **优化**: 同时更新 FusionEncoder 的 Projector 和 RDT 的 LoRA 权重。

---

## 🛠️ 安装指南

推荐使用 Conda 环境：

```bash
# 创建并激活环境
conda create -n ego_rdt python=3.10
conda activate ego_rdt

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 安装核心依赖
pip install diffusers transformers timm einops h5py opencv-python accelerate peft
```

## 🚀 训练流程
### 1. 准备统计文件
计算数据集的均值和方差，用于 Z-Score 归一化。

```Bash
python utils/compute_stats.py \
  --data_root /yanghaochuan/data/train_data.hdf5 \
  --save_path /yanghaochuan/data/dataset_stats.json
```
### 2. Stage B: 编码器预训练 (Optional but Recommended)
训练 FusionEncoder 以对齐教师特征。这一步生成的权重将作为 Stage C 的初始化。

```Bash
python train/stageB_train.py \
  --data_root /yanghaochuan/data/train_data.hdf5 \
  --output_dir /yanghaochuan/checkpoints \
  --batch_size 48 \
  --epochs 5
python train/stageB_train.py --data_root /yanghaochuan/data/1223pick_up_the_paper_cup.hdf5 --output_dir /yanghaochuan/checkpoints --batch_size 16 --epochs 5
```
### 3. Stage C: 联合训练 (Joint Training)
这是最关键的步骤。启用 Modality Dropout 和 Action Chunking。

```Bash
python train/stageC_joint.py \
  --data_root /yanghaochuan/data/train_data.hdf5 \
  --output_dir /yanghaochuan/checkpoints \
  --stage_b_ckpt /yanghaochuan/checkpoints/stageB_final.pt \
  --batch_size 16 \
  --epochs 50 \
  --pred_horizon 16
```
注意: 如果显存不足，请减小 batch_size。此阶段不再使用 Latent Cache，而是端到端训练以支持动态 Dropout。

## 🤖 推理与部署
系统采用 Client-Server 架构，支持异步非阻塞控制。

### 1. 启动推理服务 (GPU Server)
加载训练好的 FusionEncoder 和 RDT LoRA 权重。

```Bash
# Server 端
python inference/server_gpu_image.py
```
(请确保 deploy_agent_safe.py 中的 STAGE_C_PATH 指向新的 checkpoint)

### 2. 启动机器人客户端 (Robot Client)
连接机械臂与推理服务器。

```Bash
# Client 端
python inference/robot_policy_system.py
```
#### 推理特性：

Single-View Input: 仅需手腕相机图像，内部自动构造 Fake Main View。

Chunked Execution: 接收 16 步动作序列，并在执行过程中异步请求下一次推理，实现无缝连接。

Safety: 内置关节限位与平滑插值保护。

## 📝 Citation
如果你使用了本项目，请引用：

```Bash
@misc{ego_exo_rdt_2025,
  author = {Haochuan Yang},
  title = {Ego-Exo Distilled RDT: A Decoupled Diffusion Policy for VLA},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.