#!/usr/bin/env python3
"""
Transformer从零实现 - 主训练脚本
"""

# 在导入matplotlib之前设置后端
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
try:
    # 尝试不同的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("中文字体配置成功")
except:
    print("中文字体配置失败，使用默认字体")

import torch
import torch.nn as nn
import argparse
import os
import sys
import math

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# 现在导入自定义模块
try:
    from src.model import TransformerEncoder
    from src.dataset import create_vocab, TextDataset, load_tiny_shakespeare
    from src.utils import set_seed
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接导入...")
    # 如果src导入失败，尝试直接导入
    import importlib.util

    spec = importlib.util.spec_from_file_location("model", "src/model.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    TransformerEncoder = model_module.TransformerEncoder

    spec = importlib.util.spec_from_file_location("dataset", "src/dataset.py")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    create_vocab = dataset_module.create_vocab
    TextDataset = dataset_module.TextDataset
    load_tiny_shakespeare = dataset_module.load_tiny_shakespeare

    spec = importlib.util.spec_from_file_location("utils", "src/utils.py")
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    set_seed = utils_module.set_seed

from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)),
                targets.contiguous().view(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('grad_clip', 1.0)
            )
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}')

        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    targets.contiguous().view(-1)
                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        print(f"开始训练，设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config['num_epochs']):
            print(f'\nEpoch {epoch + 1}/{self.config["num_epochs"]}')
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f'训练损失: {train_loss:.4f}')
            print(f'验证损失: {val_loss:.4f}')
            print(f'学习率: {self.scheduler.get_last_lr()[0]:.6f}')

            # 保存最佳模型
            if not self.val_losses or val_loss <= min(self.val_losses):
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('保存最佳模型!')

    def plot_losses(self, save_path='training_loss.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', linewidth=2)
        plt.plot(self.val_losses, label='验证损失', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'损失曲线已保存: {save_path}')


def prepare_dataloaders(texts, vocab, config):
    """准备数据加载器"""
    seq_len = config['max_seq_len']
    batch_size = config['batch_size']

    # 将长文本分割成固定长度的序列
    sequences = []
    for text in texts:
        for i in range(0, len(text) - seq_len, seq_len):
            sequences.append(text[i:i + seq_len])

    dataset = TextDataset(sequences, vocab, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 在Windows上设置为0避免多进程问题
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_ff', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--max_seq_len', type=int, default=128, help='序列最大长度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 加载数据
    print("加载Tiny Shakespeare数据集...")
    train_text, val_text = load_tiny_shakespeare()
    print(f"训练文本长度: {len(train_text)} 字符")
    print(f"验证文本长度: {len(val_text)} 字符")

    # 创建词汇表
    vocab = create_vocab([train_text, val_text])
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")

    # 准备数据加载器
    config = {
        'max_seq_len': args.max_seq_len,
        'batch_size': args.batch_size,
        'num_workers': 0  # Windows上设置为0
    }

    train_loader = prepare_dataloaders([train_text], vocab, config)
    val_loader = prepare_dataloaders([val_text], vocab, config)

    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")

    # 初始化模型
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    )

    # 训练配置
    train_config = {
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'grad_clip': args.grad_clip,
        'weight_decay': args.weight_decay,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 创建训练器并开始训练
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train()

    # 绘制损失曲线
    trainer.plot_losses('results/training_loss.png')

    print("训练完成！")


if __name__ == '__main__':
    main()