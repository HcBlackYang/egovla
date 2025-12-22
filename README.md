# Ego-Exo Distilled RDT: Decoupled Diffusion Policy for VLA

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.29+-yellow.svg)](https://huggingface.co/docs/diffusers/index)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-blue.svg)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **核心思想**：通过双教师蒸馏（Dual-Teacher Distillation）与任务/背景解耦（Task/Background Decoupling），实现基于"纯手腕视角"（Wrist-View / Ego-Centric）的高鲁棒性机器人操控模型。

本项目是一个端到端的具身智能（Embodied AI）系统，结合了 **VideoMAEv2** 的时序感知能力与 **Robotics Diffusion Transformer (RDT)** 的动作生成能力。

---

## 🌟 主要特性

### 👁️ 纯手腕视角推理 (Ego-Centric Inference)
- 虽然训练时利用全局视角（Third-View）进行知识蒸馏，但推理时仅依赖手腕相机输入
- 解决了移动操作中第三方相机难以固定的痛点

### ⚡️ 高效训练架构 (Latent Caching + LoRA)
- **Latent Caching**: 预先提取并缓存 VideoMAE 特征，消除重复的视觉编码计算，训练速度提升 50x+
- **LoRA Fine-tuning**: 冻结 1.2B 参数的主干，仅训练 RDT 的 Low-Rank 适配器（约 3.7M 参数），显存占用大幅降低

### 🎓 双教师蒸馏架构 (Dual-Teacher Distillation)
- **语义教师 (Semantic Teacher)**: 使用冻结的 **SigLIP** (So400m) 提供强大的开放世界语义理解
- **时序/手部教师 (Temporal Teacher)**: 使用 **Exo-View** 特征（如手部视角特征）强化对动作细节的捕捉

### 🦾 扩散策略大脑 (Diffusion Policy Head)
- 集成 **RDT-1B** 作为策略头，通过 Early Fusion 将感知特征注入
- 采用 **DDIM Scheduler** 进行去噪，在保证生成质量的同时优化推理延迟

---

## 🏗️ 系统架构

系统分为三个主要阶段：

1. **Stage B (Distillation & Decoupling)**: 训练 `FusionEncoder`
   - 冻结 VideoMAE 主干的大部分层
   - 训练对齐头和解耦路由层，使其特征逼近 SigLIP 和 Exo 教师

2. **Cache Latents (Pre-computation)**: 特征缓存
   - 使用训练好的 Stage B Encoder 提取所有视频帧的 Latent 特征
   - 保存为 HDF5 格式，供 Stage C 极速读取

3. **Stage C (Latent LoRA Tuning)**: 训练 `RDT Policy`
   - 纯策略学习阶段：直接加载 Latent 特征
   - 使用 LoRA 微调 RDT 主干，实现极速收敛（~1000 samples/s）

---

## 🛠️ 安装指南

### 环境配置

推荐使用 Conda 环境：

```bash
# 创建并激活环境
conda create -n ego_rdt python=3.10
conda activate ego_rdt

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 安装核心依赖 (新增 peft)
pip install diffusers transformers timm einops h5py opencv-python accelerate peft
```

## 🚀 训练流程
### 1. Stage B: 蒸馏与解耦 (Distillation)
此阶段训练 FusionEncoder 以对齐教师特征。

```Bash
python train/stageB_train.py \
  --data_root /yanghaochuan/data/train_data.hdf5 \
  --output_dir /yanghaochuan/checkpoints \
  --batch_size 48 \
  --epochs 5
```

### 2. Cache Latents: 特征缓存
利用训练好的 Stage B 模型提取特征，生成缓存文件。

```Bash
python utils/cache_latents.py \
  --data_root /yanghaochuan/data/train_data.hdf5 \
  --stage_b_ckpt /yanghaochuan/checkpoints/stageB_final.pt \
  --output_path /yanghaochuan/data/latents_cache.hdf5
```

### 3. Stage C: LoRA 策略学习 (Latent + LoRA)
加载缓存特征，仅微调 RDT 的 LoRA 参数。

```Bash
python train/stageC_latent_lora.py \
  --cache_path /yanghaochuan/data/latents_cache.hdf5 \
  --output_dir /yanghaochuan/checkpoints \
  --batch_size 128 \
  --epochs 50
```
## 🤖 推理与部署
在线实时推理 (GPU Server)
采用 Client-Server 架构，Server 端负责重型模型推理，Client 端负责机器人控制。

启动 GPU 推理服务:

```Bash
# Server 端加载 Stage B Encoder 和 Stage C LoRA 权重
python -m inference.server_gpu_image
```
启动机械臂客户端:

```Bash
# Client 端采集图像并执行动作
python inference/robot_policy_system.py
```
推理特性：

Split Loading: 分别加载 Encoder 权重和 LoRA Policy 权重

Optimized: 启用 torch.compile 加速 Encoder，使用 DDIM Scheduler 稳定生成

Robustness: 内置 Z-Score 反归一化与动量/重力偏置策略，解决通信延迟带来的悬停问题

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