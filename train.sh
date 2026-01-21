#!/bin/bash

# =================================================================
# 🤖 EgoVLA 全流程训练脚本 (Stage B -> Stage C)
# =================================================================

# === 1. 基础环境设置 ===
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export HF_HOME=/yanghaochuan/cache/huggingface
export TORCH_HOME=/yanghaochuan/cache/torch
mkdir -p $HF_HOME
mkdir -p $TORCH_HOME

# === 2. 项目路径设置 ===
PROJECT_DIR="/yanghaochuan/projects"
cd $PROJECT_DIR
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
echo "📂 Working Directory: $(pwd)"

# === 3. 解释器与数据路径 (集中配置，方便修改) ===
PYTHON_EXE="/opt/conda/envs/ego/bin/python"
DATA_ROOT="/yanghaochuan/data/hdf5/pick_up_the_orange_ball_and_put_it_on_the_plank.hdf5"
OUTPUT_DIR="/yanghaochuan/checkpoints"
STAGE_A_CKPT="/yanghaochuan/checkpoints/stageA_final.pt"

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# =================================================================
# 🟢 第一阶段: Stage B (Representation Learning)
# =================================================================
echo "-----------------------------------------------------------"
echo "🚀 Starting Stage B Training (VideoMAE Distillation)..."
echo "-----------------------------------------------------------"

# Stage B 输出的最终模型路径 (与 stageB_train.py 代码中的保存名一致)
STAGE_B_FINAL_PATH="${OUTPUT_DIR}/120stageB_final.pt"

$PYTHON_EXE -u train/stageB_train.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --stage_a_ckpt $STAGE_A_CKPT \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 10000 \
    --checkpointing_steps 500 \
    --use_wandb

# 🛑 错误检查：如果 Stage B 失败，不要继续跑 Stage C
if [ $? -ne 0 ]; then
    echo "❌ Stage B Training Failed! Stopping pipeline."
    exit 1
fi

echo "✅ Stage B Finished successfully!"
echo "📄 Checkpoint saved at: $STAGE_B_FINAL_PATH"

# # =================================================================
# # 🔵 第二阶段: Stage C (Policy Learning)
# # =================================================================
# echo "-----------------------------------------------------------"
# echo "🚀 Starting Stage C Training (VLA Policy)..."
# echo "-----------------------------------------------------------"

# # 注意：这里 --stage_b_ckpt 自动指向了上面刚刚生成的 stageB_final.pt

# $PYTHON_EXE -u train/stageC_joint.py \
#     --data_root $DATA_ROOT \
#     --output_dir $OUTPUT_DIR \
#     --stage_b_ckpt $STAGE_B_FINAL_PATH \
#     --batch_size 32 \
#     --gradient_accumulation_steps 2 \
#     --max_train_steps 10000 \
#     --checkpointing_steps 500 \
#     --pred_horizon 64 \
#     --use_wandb

# # 🛑 错误检查
# if [ $? -ne 0 ]; then
#     echo "❌ Stage C Training Failed!"
#     exit 1
# fi

# echo "-----------------------------------------------------------"
# echo "🎉 All Stages Finished Successfully!"
# echo "-----------------------------------------------------------"