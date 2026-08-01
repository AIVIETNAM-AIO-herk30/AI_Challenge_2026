"""
Turbovec-backed vector store for video frame embeddings.

Turbovec's IdMapIndex only accepts uint64 handles, not the string frame_id
used everywhere else in the pipeline (docs/API_CONTRACT.md's working
convention: "{video_id}_{frame_index}"). This class keeps a JSON side-car
mapping frame_id <-> handle, the same pattern turbovec's own bundled
framework integrations (turbovec/langchain.py, llama_index.py) use for
exactly this problem, and validates the two stay in sync on load via
turbovec._persist.check_persisted_handles.

One store == one embedding space. SigLIP and BEiT-3 have different output
dims, so the orchestrator holds two separate TurbovecStore instances
rather than one store serving both.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from turbovec import IdMapIndex
from turbovec._persist import check_persisted_handles


class TurbovecStore:
    def __init__(self, dim: int, bit_width: int = 4):
        self.dim = dim
        self.bit_width = bit_width
        self._index = IdMapIndex(dim=dim, bit_width=bit_width)
        self._frame_id_to_handle: dict[str, int] = {}
        self._handle_to_frame_id: dict[int, str] = {}
        self._next_handle = itertools.count(1)

    def insert(self, frame_id: str, vector: np.ndarray) -> None:
        existing = self._frame_id_to_handle.get(frame_id)
        if existing is not None:
            self._index.remove(existing)
            del self._handle_to_frame_id[existing]

        handle = next(self._next_handle)
        vec = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        self._index.add_with_ids(vec, np.array([handle], dtype=np.uint64))
        self._frame_id_to_handle[frame_id] = handle
        self._handle_to_frame_id[handle] = frame_id

    def insert_batch(self, frame_ids: list[str], vectors: np.ndarray) -> None:
        if len(frame_ids) != len(vectors):
            raise ValueError("frame_ids and vectors must be the same length")
        for frame_id, vector in zip(frame_ids, vectors):
            self.insert(frame_id, vector)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        query = np.ascontiguousarray(query_vec.reshape(1, -1).astype(np.float32))
        scores, handles = self._index.search(query, top_k)

        results = []
        for score, handle in zip(scores[0], handles[0]):
            frame_id = self._handle_to_frame_id.get(int(handle))
            if frame_id is None:
                continue  # stale/unmapped handle, shouldn't happen but don't crash a search over it
            results.append((frame_id, float(score)))
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._index.write(str(path.with_suffix(".tvim")))
        sidecar = {
            "dim": self.dim,
            "bit_width": self.bit_width,
            "frame_id_to_handle": self._frame_id_to_handle,
        }
        path.with_suffix(".sidecar.json").write_text(json.dumps(sidecar))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._index = IdMapIndex.load(str(path.with_suffix(".tvim")))
        sidecar = json.loads(path.with_suffix(".sidecar.json").read_text())
        self.dim = sidecar["dim"]
        self.bit_width = sidecar["bit_width"]
        self._frame_id_to_handle = {k: int(v) for k, v in sidecar["frame_id_to_handle"].items()}

        check_persisted_handles(self._index, self._frame_id_to_handle.values(), what="frame")

        self._handle_to_frame_id = {v: k for k, v in self._frame_id_to_handle.items()}
        self._next_handle = itertools.count(max(self._frame_id_to_handle.values(), default=0) + 1)

    def __len__(self) -> int:
        return len(self._index)
