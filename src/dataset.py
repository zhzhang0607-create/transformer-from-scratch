import torch
from torch.utils.data import Dataset
import re


class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = [self.vocab.get(char, self.vocab['<unk>']) for char in text]

        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [self.vocab['<pad>']] * (self.seq_len - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


def create_vocab(texts):
    # 创建字符级词汇表
    chars = set()
    for text in texts:
        chars.update(text)

    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }

    for i, char in enumerate(sorted(chars)):
        vocab[char] = i + 4

    return vocab


def load_tiny_shakespeare():
    """加载Tiny Shakespeare数据集"""
    import requests

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text

    # 分割成训练和验证集
    n = len(text)
    train_text = text[:int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    return train_text, val_text