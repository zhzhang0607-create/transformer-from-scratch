"""
Transformer from Scratch - Source Package
"""

# Import and expose key classes
try:
    from .model import TransformerEncoder, TransformerEncoderLayer
    from .attention import MultiHeadAttention, ScaledDotProductAttention
    from .ffn import PositionWiseFFN
    from .embedding import PositionalEncoding, TokenEmbedding
    from .dataset import TextDataset, create_vocab, load_tiny_shakespeare
    from .utils import set_seed, count_parameters, create_mask
    # 移除 ensure_dir 如果它不存在
except ImportError as e:
    # If relative imports fail, this might be running as main script
    print(f"Import warning: {e}")
    pass

__all__ = [
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    'PositionWiseFFN',
    'PositionalEncoding',
    'TokenEmbedding',
    'TextDataset',
    'create_vocab',
    'load_tiny_shakespeare',
    'set_seed',
    'count_parameters',
    'create_mask'
]