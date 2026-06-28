"""
Video frame sampling and indexing pipeline.
Owner: Pham Viet Truong
"""

from pathlib import Path


class VideoIndexer:
    """
    Walks a directory of .mp4 files, samples frames at a given FPS,
    computes SigLIP embeddings via VisualAgent, and loads them into VectorStore.

    TODO (Pham Viet Truong):
    - Implement index_directory(video_dir) to iterate all .mp4 files
    - Use decord.VideoReader for fast frame sampling
    - Batch frames and call visual_agent.process() concurrently
    - Call vector_store.train() once before first add()
    """

    def __init__(self, visual_agent, vector_store, fps: float = 1.0):
        self._agent = visual_agent
        self._store = vector_store
        self._fps = fps

    async def index_directory(self, video_dir: str | Path) -> None:
        raise NotImplementedError
