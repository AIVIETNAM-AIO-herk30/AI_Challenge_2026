"""
FAISS vector store for video frame embeddings.
Owner: Pham Viet Truong
"""

from pathlib import Path

import numpy as np


class VectorStore:
    """
    TODO (Pham Viet Truong):
    - Initialize FAISS IVFFlat index in __init__ (use faiss-gpu if available)
    - Implement train(embeddings) — required before first add()
    - Implement add(embeddings, metadata) — metadata = list of dicts
      with keys: video_id, frame_idx, timestamp_sec
    - Implement search(query_vec, top_k) → list of metadata dicts with "score"
    - Implement save(path) and load(path)
    """

    def __init__(self, embed_dim: int = 1152, nlist: int = 256, nprobe: int = 32):
        self.embed_dim = embed_dim
        self.nprobe = nprobe
        # TODO: create FAISS index

    def train(self, embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[dict]:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
