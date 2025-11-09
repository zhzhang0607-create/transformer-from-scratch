#!/bin/bash

# 设置环境
echo "设置训练环境..."

# 创建结果目录
mkdir -p results

# 安装依赖
pip install -r requirements.txt

# 设置随机种子
SEED=42

# 运行训练
echo "开始训练Transformer模型..."
python train.py \
    --d_model 128 \
    --num_heads 4 \
    --num_layers 2 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --max_seq_len 128 \
    --num_epochs 50 \
    --seed $SEED

echo "训练完成！结果保存在 results/ 目录中"

# 运行消融实验
echo "开始消融实验..."
python ablation_study.py --seed $SEED

echo "所有实验完成！"