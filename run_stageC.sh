#!/bin/bash

# === 1. 设置环境变量，防止编码错误 ===
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# === 2. 显卡设置 (单卡直接跑，如果多卡请指定 ID) ===
# export CUDA_VISIBLE_DEVICES=0 

# === 3. 重定向缓存目录 (防止无权限或系统盘爆满) ===
export HF_HOME=/yanghaochuan/cache/huggingface
export TORCH_HOME=/yanghaochuan/cache/torch
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# === 4. 强制切换到项目目录 ===
PROJECT_DIR="/yanghaochuan/projects" # <--- 请确认这是你项目的根目录
cd $PROJECT_DIR
echo "Current working directory: $(pwd)"

# === 5. 设置 PYTHONPATH (确保能 import model 和 utils) ===
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

# === 6. Python解释器路径 ===
# 请确认这是你环境里 `which python` 的结果
PYTHON_EXE="/opt/conda/envs/ego/bin/python" 

# === 7. 运行命令 (A800 优化版参数) ===
# 修改点：
# 1. 脚本改为 stageC_step_based.py
# 2. Batch Size = 32 (A800 大显存优化)
# 3. Acc Steps = 2 (等效 64)
# 4. Max Steps = 3000 (快速收敛策略)

echo "Starting training..."

$PYTHON_EXE -u train/stageC_joint.py \
    --data_root /yanghaochuan/data/1223pick_up_the_paper_cup.hdf5 \
    --output_dir /yanghaochuan/checkpoints \
    --stage_b_ckpt /yanghaochuan/checkpoints/1223stageB_papercup.pt \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 10000 \
    --checkpointing_steps 200 \
    --pred_horizon 64 \
    --use_wandb

echo "Training finished."