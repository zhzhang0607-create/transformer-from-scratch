# Transformer from Scratch

一个从零开始实现的Transformer模型，专为语言建模任务设计。本项目完整实现了Transformer编码器架构，包含多头自注意力机制、位置编码、前馈网络等核心组件，并提供了完整的训练框架和消融实验。

## ✨ 项目特点

- ✅ **完整实现**: Multi-Head Self-Attention, Position-wise FFN, 残差连接, Layer Normalization
- ✅ **多种位置编码**: 正弦位置编码
- ✅ **训练优化**: 学习率调度、梯度裁剪、AdamW优化器
- ✅ **可视化**: 训练曲线、消融实验结果
- ✅ **模块化设计**: 易于理解和扩展的代码结构
- ✅ **完整文档**: 详细的配置说明和复现指南

## 📋 作业要求完成情况

| 要求 | 完成状态 | 说明 |
|------|----------|------|
| 报告撰写 | ✅ | 包含完整数学推导、伪代码、实验分析 |
| Multi-Head Self-Attention | ✅ | 完整实现，支持掩码 |
| Position-wise FFN | ✅ | 两层MLP，GELU激活 |
| 残差连接 + LayerNorm | ✅ | SublayerConnection模块 |
| 位置编码 | ✅ | 正弦位置编码 |
| 代码开源 | ✅ | 完整GitHub仓库结构 |
| 训练框架 | ✅ | 完整训练循环、验证、保存 |
| 消融实验 | ✅ | 5组对比实验 |
| 可视化 | ✅ | 损失曲线、实验结果图 |
| Encoder实现 | ✅ | 完整Transformer编码器架构 |

## 🚀 快速开始

### 环境要求

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **内存**: 至少8GB RAM
- **存储**: 至少2GB可用空间
- **GPU** (可选): 4GB+ VRAM用于加速训练

### 安装依赖

```bash
# 创建conda环境（推荐）
conda create -n transformer python=3.10
conda activate transformer

# 安装PyTorch（根据您的CUDA版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或CPU版本
pip install torch torchvision torchaudio

# 安装项目依赖
pip install -r requirements.txt
```

### 快速训练

```bash
# 使用默认配置训练（5个epoch快速测试）
python train.py --num_epochs 5

# 完整训练（10个epoch）
python train.py --seed 42
```

### 消融实验

```bash
# 运行完整消融实验（5个配置，各5个epoch）
python ablation_study.py --seed 42

# 快速测试版本
python ablation_study.py --num_epochs 2 --seed 42
```

## 📁 项目结构

```
transformer-from-scratch/
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── model.py           # Transformer编码器模型
│   ├── attention.py       # 注意力机制实现
│   ├── ffn.py            # 前馈网络实现
│   ├── embedding.py       # 词嵌入和位置编码
│   ├── dataset.py         # 数据集加载和处理
│   └── utils.py          # 工具函数
├── configs/               # 配置文件目录
│   └── base.yaml         # 基础训练配置
├── scripts/               # 运行脚本目录
│   └── run.sh            # 自动化运行脚本
├── results/               # 实验结果目录（自动生成）
│   ├── training_loss.png
│   ├── ablation_results.png
│   └── ablation_details.json
├── requirements.txt       # Python依赖列表
├── train.py              # 主训练脚本
├── ablation_study.py     # 消融实验脚本
├── README.md             # 项目说明文档
└── .gitignore           # Git忽略文件
```

## ⚙️ 模型架构

### 核心组件

**Multi-Head Self-Attention**
```python
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Position-wise Feed-Forward Network**
```python
FFN(x) = GELU(xW1 + b1)W2 + b2
```

**位置编码**
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**残差连接和LayerNorm**
```python
Output = LayerNorm(x + Sublayer(x))
```

### 默认超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 128 | 模型维度 |
| num_heads | 4 | 注意力头数 |
| num_layers | 2 | Transformer层数 |
| d_ff | 512 | 前馈网络维度 |
| max_seq_len | 128 | 最大序列长度 |
| dropout | 0.1 | Dropout率 |
| batch_size | 32 | 批次大小 |
| learning_rate | 3e-4 | 学习率 |

## 🔬 实验设置

### 数据集

使用 **Tiny Shakespeare** 数据集进行字符级语言建模：
- **训练集**: 1,003,854字符 (90%)
- **验证集**: 111,540字符 (10%) 
- **词汇表大小**: 69个字符
- **自动下载**: 代码包含数据集下载功能

### 评估指标

- **交叉熵损失**: 主要训练目标
- **困惑度**: exp(loss)
- **训练稳定性**: 损失曲线平滑度

### 消融实验设计

1. **baseline**: 标准配置 (4头, 2层, 128维)
2. **2_heads**: 减少注意力头数至2
3. **8_heads**: 增加注意力头数至8  
4. **small_model**: 减小模型规模 (64维, 256 FFN)
5. **single_layer**: 减少编码器层数至1

## 📊 精确复现

### 完整训练命令

```bash
python train.py \
    --d_model 128 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 512 \
    --max_seq_len 128 \
    --dropout 0.1 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --num_epochs 10 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --seed 42
```

### 预期结果

- **训练时间**: CPU约2-4小时，GPU约30-60分钟
- **最终验证损失**: 约2.45-2.50
- **模型大小**: 约1.6MB (414,533参数)
- **训练曲线**: 持续收敛，无过拟合

## 📈 实验结果

### 消融实验结果

| 模型配置 | 参数量 | 最终验证损失 | 相对基线变化 |
|----------|--------|--------------|--------------|
| baseline | 414,533 | 2.4773 | - |
| 2_heads | 414,533 | 2.4771 | -0.0002 |
| 8_heads | 414,533 | **2.4682** | **-0.0091** |
| small_model | 108,997 | 2.5042 | +0.0269 |
| single_layer | 216,261 | 2.4837 | +0.0064 |

### 结果分析

- **8头注意力表现最佳**，验证损失2.4682
- **模型容量很重要**，small_model性能明显下降
- **多层结构有帮助**，single_layer性能略差于baseline

## 🔧 核心代码

### 多头注意力实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        # 线性变换并分头
        Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output), attn_weights
```

### 位置编码实现

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)
```

## 🐛 故障排除

### 常见问题

**内存不足**
```bash
# 减小批次大小
python train.py --batch_size 16

# 减小序列长度
python train.py --max_seq_len 64
```

**训练不收敛**
```bash
# 调整学习率
python train.py --learning_rate 1e-4

# 检查梯度
python train.py --grad_clip 0.5
```

**导入错误**
```bash
# 确保在项目根目录运行
cd transformer-from-scratch

# 检查Python路径
python -c "import src; print('导入成功')"
```

### 调试模式

```bash
# 快速功能测试
python train.py --num_epochs 1 --batch_size 8
python ablation_study.py --num_epochs 1
```

## 🔮 扩展开发

### 添加新模块

1. 在对应文件中实现新模块
2. 在 `src/__init__.py` 中导出
3. 在模型架构中集成

### 自定义数据集

修改 `src/dataset.py`：

```python
def load_custom_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 自定义数据处理
    return train_text, val_text
```

### 扩展实验

修改 `ablation_study.py` 添加新实验：

```python
new_experiment = {
    'name': 'large_model',
    'd_model': 256,
    'num_heads': 8,
    'd_ff': 1024,
    'num_layers': 4
}
```

## 📝 学术引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{transformer_scratch_2024,
  title = {Transformer from Scratch Implementation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/transformer-from-scratch}
}
```

## 🤝 贡献指南

欢迎贡献代码和提出问题！

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [Vaswani et al.](https://arxiv.org/abs/1706.03762) 的原始Transformer论文
- 感谢 PyTorch 团队提供的优秀框架
- 感谢开源社区的贡献和支持

---

**注意**: 本项目为教育目的设计，适合学习和研究使用。生产环境建议使用优化库如 [Hugging Face Transformers](https://github.com/huggingface/transformers)。