# Ego-Exo Distilled RDT: Decoupled Diffusion Policy for VLA

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.29+-yellow.svg)](https://huggingface.co/docs/diffusers/index)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **核心思想**：通过双教师蒸馏（Dual-Teacher Distillation）与任务/背景解耦（Task/Background Decoupling），实现基于"纯手腕视角"（Wrist-View / Ego-Centric）的高鲁棒性机器人操控模型。

本项目是一个端到端的具身智能（Embodied AI）系统，结合了 **VideoMAEv2** 的时序感知能力与 **Robotics Diffusion Transformer (RDT)** 的动作生成能力。

---

## 🌟 主要特性

### 👁️ 纯手腕视角推理 (Ego-Centric Inference)
- 虽然训练时利用全局视角（Third-View）进行知识蒸馏，但推理时仅依赖手腕相机输入
- 解决了移动操作中第三方相机难以固定的痛点

### 🎓 双教师蒸馏架构 (Dual-Teacher Distillation)
- **语义教师 (Semantic Teacher)**: 使用冻结的 **SigLIP** (So400m) 提供强大的开放世界语义理解
- **时序/手部教师 (Temporal Teacher)**: 使用 **Exo-View** 特征（如手部视角特征）强化对动作细节的捕捉

### 🧩 任务/背景解耦 (Decoupled Representation)
- 引入 `DecouplingLoss` 和 `InvarianceLoss`，强迫模型将特征分离为"任务相关槽（Task Slots）"和"背景上下文（Background Context）"
- 显著提升在复杂、动态背景下的抗干扰能力

### 🦾 扩散策略大脑 (Diffusion Policy Head)
- 集成 **RDT-1B** 作为策略头，通过 Early Fusion 将感知特征注入
- 移除冗余的状态 Token，实现更平滑、拟人的动作生成

---

## 🏗️ 系统架构

系统分为三个主要训练阶段：

1. **Stage A (Optional)**: 纯重建预训练
   - 本项目跳过此阶段，直接利用 VideoMAE 预训练权重

2. **Stage B (Distillation & Decoupling)**: 训练 `FusionEncoder`
   - 冻结 VideoMAE 主干的大部分层
   - 训练对齐头和解耦路由层
   - 使其特征逼近 SigLIP 和 Exo 教师

3. **Stage C (Joint Tuning)**: 训练 `RDT Policy`
   - 冻结感知部分
   - 训练 RDT 根据融合特征 `e_t` 生成动作

---

## 🛠️ 安装指南

### 环境配置

推荐使用 Conda 环境：

```bash
# 创建并激活环境
conda create -n ego_rdt python=3.10
conda activate ego_rdt

# 安装 PyTorch (根据你的 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
pip install diffusers transformers timm einops h5py opencv-python accelerate
```

---

## 📂 数据准备

### 数据格式

数据需预处理为 HDF5 格式。本项目包含自动提取教师特征的脚本。

**数据结构要求：**

```
data.hdf5
├── data
│   ├── demo_0
│   │   ├── obs
│   │   │   ├── robot0_eye_in_hand_image  # [Source: wrist_image.mp4] 核心输入 (Student Input)
│   │   │   ├── agentview_image           # [Source: main_image.mp4] 全局视角 (Teacher Input)
│   │   │   └── robot0_joint_pos          # [Source: FrankaEmika_states.json] 机械臂状态
│   │   ├── teacher_siglip                # 基于 agentview_image 预提取的语义特征
│   │   ├── teacher_exo                   # 基于 robot0_eye_in_hand_image 预提取的时序特征
│   │   ├── actions                       # [Source: FrankaEmika_states.json] 动作真值
│   │   └── attrs: language_instruction   # [Source: task_info.json] 语言指令
│   ├── demo_1 ...
```

### 预处理脚本

运行以下命令提取教师特征：

```bash
python utils/preprocess_with_teachers.py \
  --raw_dir /yanghaochuan/projects/data/ego \
  --out_path /yanghaochuan/projects/data/train_data.hdf5 \
  --siglip_path /yanghaochuan/models/siglip-so400m-patch14-384
```

---

## 🚀 训练流程

### Stage B: 蒸馏与解耦 (Distillation)

此阶段训练 FusionEncoder 以对齐教师特征。

```bash
python train/stageB_train.py \
  --data_root /yanghaochuan/projects/data/train_data.hdf5 \
  --output_dir /yanghaochuan/projects/checkpoints \
  --batch_size 48 \
  --epochs 5
```

### Stage C: 策略学习 (Policy Learning)

此阶段加载 Stage B 的权重，训练 RDT 扩散模型。

```bash
python train/stageC_joint.py \
  --data_root /yanghaochuan/projects/data/train_data.hdf5 \
  --stage_b_ckpt /yanghaochuan/projects/checkpoints/stageB_final.pt \
  --output_dir /yanghaochuan/projects/checkpoints \
  --batch_size 48 \
  --epochs 5
```

---

## 🤖 推理与部署

### 在线实时推理 (Socket Server)

启动推理服务，等待机械臂客户端连接：

```bash
python inference/infer_loop.py
```

**功能说明：**
- **输入**: 手腕摄像头实时画面 + 机械臂状态
- **输出**: 7DoF 关节动作
- **特性**: 使用 DPMSolver 进行 10 步快速采样

### 模型导出 (ONNX)

用于边缘端部署加速：

```bash
python inference/export_onnx.py --weights /yanghaochuan/projects/checkpoints/stageC_final.pt
```



---

## 📝 Citation

如果你使用了本项目，请引用：

```bibtex
@misc{ego_exo_rdt_2025,
  author = {Haochuan Yang},
  title = {Ego-Exo Distilled RDT: A Decoupled Diffusion Policy for VLA},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.