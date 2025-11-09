#!/usr/bin/env python3
"""
Ablation Study Script
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import custom modules
try:
    from src.model import TransformerEncoder
    from src.dataset import create_vocab, TextDataset, load_tiny_shakespeare
    from src.utils import set_seed
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct import...")
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


# Define Trainer class here since it's not in src
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
        print(f"开始训练, 设备: {self.device}")
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

            # Save best model
            if not self.val_losses or val_loss <= min(self.val_losses):
                torch.save(self.model.state_dict(), f'best_model_{self.config.get("exp_name", "default")}.pth')
                print('保存最佳模型!')

        return {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'train_losses': self.train_losses.copy(),
            'val_losses': self.val_losses.copy()
        }


def prepare_dataloaders(texts, vocab, config):
    """Prepare data loaders"""
    seq_len = config['max_seq_len']
    batch_size = config['batch_size']

    # Split long texts into fixed-length sequences
    sequences = []
    for text in texts:
        for i in range(0, len(text) - seq_len, seq_len):
            sequences.append(text[i:i + seq_len])

    dataset = TextDataset(sequences, vocab, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 on Windows
    )

    return dataloader


def run_ablation_experiment(exp_config, train_text, val_text, vocab):
    """Run a single ablation experiment"""
    print(f"\n运行试验: {exp_config['name']}")

    # Prepare data loaders
    data_config = {
        'max_seq_len': exp_config['max_seq_len'],
        'batch_size': exp_config['batch_size'],
        'num_workers': 0
    }

    train_loader = prepare_dataloaders([train_text], vocab, data_config)
    val_loader = prepare_dataloaders([val_text], vocab, data_config)

    # Initialize model
    model = TransformerEncoder(
        vocab_size=len(vocab),
        d_model=exp_config['d_model'],
        num_heads=exp_config['num_heads'],
        d_ff=exp_config['d_ff'],
        num_layers=exp_config['num_layers'],
        max_seq_len=exp_config['max_seq_len'],
        dropout=exp_config.get('dropout', 0.1)
    )

    # Training configuration
    train_config = {
        'learning_rate': exp_config['learning_rate'],
        'num_epochs': exp_config.get('num_epochs', 10),  # Use fewer epochs for ablation
        'grad_clip': exp_config.get('grad_clip', 1.0),
        'weight_decay': exp_config.get('weight_decay', 0.01),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'exp_name': exp_config['name']
    }

    # Train
    trainer = Trainer(model, train_loader, val_loader, train_config)
    result = trainer.train()

    return result


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs per experiment')
    args = parser.parse_args()

    set_seed(args.seed)

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Load data
    print("加载数据...")
    train_text, val_text = load_tiny_shakespeare()
    vocab = create_vocab([train_text, val_text])

    print(f"训练文本长度: {len(train_text)} 字符")
    print(f"验证文本长度: {len(val_text)} 字符")
    print(f"词汇表大小: {len(vocab)}")

    # Ablation experiment configurations
    experiments = [
        {
            'name': 'baseline',
            'd_model': 128,
            'num_heads': 4,
            'd_ff': 512,
            'num_layers': 2,
            'max_seq_len': 128,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': args.num_epochs
        },
        {
            'name': '2_heads',
            'd_model': 128,
            'num_heads': 2,
            'd_ff': 512,
            'num_layers': 2,
            'max_seq_len': 128,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': args.num_epochs
        },
        {
            'name': '8_heads',
            'd_model': 128,
            'num_heads': 8,
            'd_ff': 512,
            'num_layers': 2,
            'max_seq_len': 128,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': args.num_epochs
        },
        {
            'name': 'small_model',
            'd_model': 64,
            'num_heads': 4,
            'd_ff': 256,
            'num_layers': 2,
            'max_seq_len': 128,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': args.num_epochs
        },
        {
            'name': 'single_layer',
            'd_model': 128,
            'num_heads': 4,
            'd_ff': 512,
            'num_layers': 1,
            'max_seq_len': 128,
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': args.num_epochs
        }
    ]

    results = {}

    # Run all experiments
    for exp_config in experiments:
        result = run_ablation_experiment(exp_config, train_text, val_text, vocab)
        results[exp_config['name']] = result

    # Plot ablation results
    plt.figure(figsize=(12, 8))

    for exp_name, result in results.items():
        plt.plot(result['val_losses'], label=exp_name, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Ablation Study - Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print final results
    print("\nAblation Study Final Results:")
    print("=" * 50)
    for exp_name, result in results.items():
        print(f"{exp_name:15} | Final Val Loss: {result['final_val_loss']:.4f}")

    # Save detailed results
    import json
    with open('results/ablation_details.json', 'w') as f:
        # Convert tensors to floats for JSON serialization
        serializable_results = {}
        for exp_name, result in results.items():
            serializable_results[exp_name] = {
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']),
                'train_losses': [float(x) for x in result['train_losses']],
                'val_losses': [float(x) for x in result['val_losses']]
            }
        json.dump(serializable_results, f, indent=2)

    print("\nAblation study completed! Results saved in results/ directory")


if __name__ == '__main__':
    main()