"""
Elasticsearch-backed text store for per-frame ASR/OCR text.

Schema (docs/API_CONTRACT.md's working convention, kept as the default):
  {"frame_id": str, "video_id": str, "timestamp_seconds": float,
   "ocr_text": str, "asr_text": str}

frame_id is used as the ES document _id (not just a field) — reruns of the
indexer overwrite in place instead of accumulating duplicates, and it makes
handoff verification trivial: GET /{index}/_doc/{frame_id}.
"""

import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan


class ElasticsearchStore:
    MAPPING = {
        "properties": {
            "frame_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "timestamp_seconds": {"type": "float"},
            "ocr_text": {"type": "text"},
            "asr_text": {"type": "text"},
        }
    }

    def __init__(self, url: str | None = None, index_name: str = "aic2026_frames"):
        self._url = url or os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
        self._client = Elasticsearch(self._url)
        self._index_name = index_name

    def ensure_index(self) -> None:
        if not self._client.indices.exists(index=self._index_name):
            self._client.indices.create(index=self._index_name, mappings=self.MAPPING)

    def upsert_frame(
        self,
        frame_id: str,
        video_id: str,
        timestamp_seconds: float,
        ocr_text: str,
        asr_text: str,
    ) -> None:
        self._client.index(
            index=self._index_name,
            id=frame_id,
            document={
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp_seconds": timestamp_seconds,
                "ocr_text": ocr_text,
                "asr_text": asr_text,
            },
        )

    def bulk_upsert(self, docs: list[dict]) -> None:
        if not docs:
            return
        actions = [
            {"_index": self._index_name, "_id": doc["frame_id"], "_source": doc}
            for doc in docs
        ]
        bulk(self._client, actions)

    def get_by_frame_id(self, frame_id: str) -> dict | None:
        try:
            result = self._client.get(index=self._index_name, id=frame_id)
        except Exception:
            return None
        return result["_source"] if result.get("found") else None

    def get_many_by_frame_ids(self, frame_ids: list[str]) -> dict[str, dict]:
        """Fetch frame metadata in one request, keyed by ``frame_id``.

        Vector retrieval returns frame IDs while Elasticsearch owns the
        user-facing metadata (video ID and timestamp).  Keeping this lookup
        batched avoids one network request per candidate.
        """
        if not frame_ids:
            return {}
        result = self._client.mget(index=self._index_name, ids=frame_ids)
        return {
            doc["_id"]: doc["_source"]
            for doc in result["docs"]
            if doc.get("found") and "_source" in doc
        }

    def search_text(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return BM25 matches across OCR and ASR text, ranked by ES score."""
        if not query.strip() or top_k < 1:
            return []
        result = self._client.search(
            index=self._index_name,
            size=top_k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["ocr_text^2", "asr_text"],
                    "type": "best_fields",
                }
            },
        )
        return [
            (hit["_id"], float(hit.get("_score") or 0.0))
            for hit in result["hits"]["hits"]
        ]

    def scan_all(self) -> list[dict]:
        """All indexed docs — fine at mini-set scale (a handful of videos).
        Used by scripts/verify_index.py to sample frame_ids per video without
        needing the in-memory IndexReport from the indexing run that produced
        them; this is Team 1's independently-checkable source of truth."""
        self._client.indices.refresh(index=self._index_name)
        return [hit["_source"] for hit in scan(self._client, index=self._index_name, query={"query": {"match_all": {}}})]

    def __len__(self) -> int:
        self._client.indices.refresh(index=self._index_name)
        return self._client.count(index=self._index_name)["count"]
