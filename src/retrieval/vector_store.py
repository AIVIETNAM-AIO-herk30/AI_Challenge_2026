"""
FAISS vector store for video frame embeddings.
Owner: Pham Viet Truong

Phase 1 (docs/IMPLEMENTATION_PLAN.md §3, §7): brute-force exact search via
IndexFlatIP wrapped in IndexIDMap, so embedding_id stays explicit and
stable — no train() step required for this index type, which removes a
whole class of "forgot to train before add" bugs. Swap in IndexIVFFlat
(Phase 2, config.yaml already has nlist/nprobe pinned) once corpus size
makes brute-force too slow; the IndexIDMap wrapper and the rest of this
class's interface don't need to change.

Metadata schema (docs/IMPLEMENTATION_PLAN.md §7.2) — one row per keyframe,
keyed by embedding_id, which must match the FAISS id exactly:
  embedding_id, video_id, frame_idx, timestamp_sec, keyframe_path,
  asr_text, ocr_text, source_type
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd


class VectorStore:
    METADATA_COLUMNS = [
        "embedding_id", "video_id", "frame_idx", "timestamp_sec",
        "keyframe_path", "asr_text", "ocr_text", "source_type",
    ]

    def __init__(self, embed_dim: int = 1152, nlist: int = 256, nprobe: int = 32):
        self.embed_dim = embed_dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(embed_dim))
        self.metadata = (
            pd.DataFrame(columns=self.METADATA_COLUMNS)
            .astype({"embedding_id": "int64"})
            .set_index("embedding_id", drop=False)
        )

    def train(self, embeddings: np.ndarray) -> None:
        """No-op for IndexFlatIP (Phase 1). Kept so callers written against
        the Phase 2 IndexIVFFlat upgrade path don't need to change."""
        return

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must be the same length")
        if len(embeddings) == 0:
            return

        vecs = np.ascontiguousarray(embeddings.astype(np.float32))
        faiss.normalize_L2(vecs)
        ids = np.array([m["embedding_id"] for m in metadata], dtype=np.int64)
        self.index.add_with_ids(vecs, ids)

        new_rows = pd.DataFrame(metadata).astype({"embedding_id": "int64"}).set_index(
            "embedding_id", drop=False
        )
        self.metadata = pd.concat([self.metadata, new_rows])

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[dict]:
        query_vec = np.ascontiguousarray(query_vec.reshape(1, -1).astype(np.float32))
        faiss.normalize_L2(query_vec)
        scores, ids = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            row = self.metadata.loc[int(idx)]
            results.append(
                {
                    "video_id": str(row["video_id"]),
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp_sec": float(row["timestamp_sec"]),
                    "score": float(score),
                }
            )
        return results

    def get_text_fields(self, video_id: str, frame_idx: int) -> dict[str, str | None]:
        """
        Used by the Phase 1 optional hybrid rerank (docs/IMPLEMENTATION_PLAN.md
        §6) to fetch stored asr_text/ocr_text for a candidate hit without
        widening the frozen search() return shape (§7.3).
        """
        match = self.metadata[
            (self.metadata["video_id"] == video_id) & (self.metadata["frame_idx"] == frame_idx)
        ]
        if match.empty:
            return {"asr_text": None, "ocr_text": None}
        row = match.iloc[0]
        return {"asr_text": row.get("asr_text"), "ocr_text": row.get("ocr_text")}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        self.metadata.to_parquet(path / "metadata.parquet", index=False)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.metadata = pd.read_parquet(path / "metadata.parquet").set_index(
            "embedding_id", drop=False
        )

    def __len__(self) -> int:
        return self.index.ntotal
