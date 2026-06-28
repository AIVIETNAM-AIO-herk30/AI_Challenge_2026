"""
Data pipeline and preprocessing.
Owner: Pham Huu Huy
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class QueryDataset(Dataset):
    """
    Dataset for training the query complexity classifier.

    Expected JSON format:
    [{"query": str, "embedding": list[float], "query_type": int, "complexity": int}, ...]

    TODO (Pham Huu Huy):
    - Implement __init__ to load and parse the JSON file
    - Implement __len__ and __getitem__
    - Add data augmentation if needed
    """

    def __init__(self, data_path: str | Path):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def build_dataloaders(
    data_path: str | Path,
    val_split: float = 0.1,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """
    TODO (Pham Huu Huy):
    - Split dataset into train/val
    - Return two DataLoaders
    """
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: add argparse for preprocessing CLI
    pass
