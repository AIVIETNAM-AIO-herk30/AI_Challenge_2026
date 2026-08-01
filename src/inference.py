"""End-to-end multimodal retrieval for the active Turbovec + Elasticsearch stack.

At query time, a text query is embedded by SigLIP and searched against the
offline-built Turbovec index.  The same query is sent to Elasticsearch for
BM25 matches over OCR and ASR text.  Reciprocal-rank fusion combines the two
rankings, then Elasticsearch hydrates frame IDs into the public result schema.

OCR and ASR agents are indexing-time tools: they consume images and audio, not
text queries.  They are therefore deliberately not instantiated here.
"""

import argparse
import asyncio

import yaml

from .agents.visual_agent import VisualAgent
from .retrieval.es_store import ElasticsearchStore
from .retrieval.vector_store import TurbovecStore

# Process-lifetime cache so repeated search() calls (e.g. from the Part 2
# eval harness, which calls this once per query) don't reload SigLIP /
# Whisper / the FAISS index from scratch every time. Keyed by the config
# dict's identity, which stays stable as long as the caller loads
# config.yaml once and reuses the dict, exactly how eval.py and the CLI
# below both use it.
_CONTEXT_CACHE: dict[int, tuple[VisualAgent, TurbovecStore, ElasticsearchStore]] = {}


def _get_context(config: dict) -> tuple[VisualAgent, TurbovecStore, ElasticsearchStore]:
    key = id(config)
    if key not in _CONTEXT_CACHE:
        visual_cfg = config["agents"]["visual"]["siglip"]
        visual_agent = VisualAgent(
            model_name=visual_cfg["model"],
            pretrained=visual_cfg["pretrained"],
            max_concurrent=visual_cfg.get("max_concurrent", 8),
        )
        vector_cfg = config["turbovec"]
        store = TurbovecStore(
            dim=visual_cfg["embed_dim"], bit_width=vector_cfg.get("bit_width", 4)
        )
        store.load(f"{vector_cfg['index_dir'].rstrip('/')}/siglip")
        es_cfg = config["elasticsearch"]
        text_store = ElasticsearchStore(
            url=es_cfg.get("url"), index_name=es_cfg["index_name"]
        )
        _CONTEXT_CACHE[key] = (visual_agent, store, text_store)
    return _CONTEXT_CACHE[key]


def _reciprocal_rank_fusion(
    *rankings: list[tuple[str, float]], constant: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked frame-ID lists without comparing incompatible score scales."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (frame_id, _score) in enumerate(ranking, start=1):
            fused[frame_id] = fused.get(frame_id, 0.0) + 1.0 / (constant + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


async def _search_async(query: str, config: dict, top_k: int) -> list[dict]:
    if top_k < 1:
        return []

    visual_agent, vector_store, text_store = _get_context(config)
    visual_result = await visual_agent.process({"text": query})
    if not visual_result.success:
        raise RuntimeError(f"query embedding failed: {visual_result.error}")

    fetch_k = top_k * 5
    vector_hits = vector_store.search(visual_result.output, top_k=fetch_k)
    text_hits = text_store.search_text(query, top_k=fetch_k)
    fused_hits = _reciprocal_rank_fusion(vector_hits, text_hits)
    metadata_by_frame_id = text_store.get_many_by_frame_ids(
        [frame_id for frame_id, _score in fused_hits[:fetch_k]]
    )

    results = []
    for frame_id, score in fused_hits:
        metadata = metadata_by_frame_id.get(frame_id)
        if metadata is None:
            continue
        results.append(
            {
                "video_id": metadata["video_id"],
                "frame_idx": int(frame_id.rsplit("_", 1)[-1]),
                "timestamp_sec": float(metadata["timestamp_seconds"]),
                "score": score,
            }
        )
        if len(results) == top_k:
            break
    return results


def search(query: str, config: dict, top_k: int = 10) -> list[dict]:
    """
    Returns: list of {"video_id": str, "frame_idx": int, "timestamp_sec": float, "score": float}
    """
    return asyncio.run(_search_async(query, config, top_k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = search(args.query, cfg, args.top_k)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")
