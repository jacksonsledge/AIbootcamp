"""
Character-level tokenizer and dataset utilities.
"""

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """Maps characters <-> integer ids."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self._c2i = {c: i for i, c in enumerate(chars)}
        self._i2c = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self._c2i[c] for c in text]

    def decode(self, ids):
        return "".join(self._i2c[i] for i in ids)


class TextDataset(Dataset):
    """Sliding-window dataset that yields (input, target) token pairs."""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_data(text: str, block_size: int, train_split: float = 0.9):
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * train_split)
    train_ds = TextDataset(data[:n], block_size)
    val_ds = TextDataset(data[n:], block_size)
    return tokenizer, train_ds, val_ds
