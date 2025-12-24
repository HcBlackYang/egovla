#!/bin/bash

# === 1. 设置环境变量，防止编码错误 ===
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# === 2. 只有在非 SSH 环境下才需要处理显卡可见性 (可选) ===
# export CUDA_VISIBLE_DEVICES=0 

# === 3. 重定向缓存目录 (防止无权限) ===
export HF_HOME=/yanghaochuan/cache/huggingface
export TORCH_HOME=/yanghaochuan/cache/torch
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# === 4. 强制切换到项目目录 ===
PROJECT_DIR="/yanghaochuan/projects" # <--- 改成你存放代码的真实路径
cd $PROJECT_DIR
echo "Current working directory: $(pwd)"

# === 5. 设置 PYTHONPATH ===
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

# === 6. 激活环境 (根据你的服务器情况，二选一) ===
# 方式 A: 如果是 Slurm/Conda
# source /path/to/anaconda3/bin/activate aloha

# 方式 B: 直接使用绝对路径的 Python (推荐，最稳)
PYTHON_EXE="/opt/conda/envs/ego/bin/python" # <--- 改成你 SSH 时 `which python` 看到的路径

# === 7. 运行命令 (加上 -u) ===
echo "Starting training..."
$PYTHON_EXE -u train/stageC_joint.py \
    --data_root /yanghaochuan/data/1223pick_up_the_paper_cup.hdf5 \
    --output_dir /yanghaochuan/checkpoints \
    --stage_b_ckpt /yanghaochuan/checkpoints/1223stageB_papercup.pt \
    --batch_size 16 \
    --epochs 50 \
    --pred_horizon 64 \
    --use_wandb  # 如果你想开 wandb，记得加上 API KEY 环境变量

echo "Training finished."