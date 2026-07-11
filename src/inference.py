"""
End-to-end inference: query -> classify -> dispatch -> retrieve video frames.
Owner: Truong Hoang Thong

Phase 1 (docs/IMPLEMENTATION_PLAN.md §6): classify the query with the
existing rule-based classifier, embed it via VisualAgent (text mode)
through the dispatcher, search the FAISS index built by Part 1, and — only
for queries the classifier flags as OCR/ASR/HYBRID — rerank the top
candidates against the *stored* asr_text/ocr_text metadata with BM25.

We deliberately do NOT call live OCR/ASR agents on the query text itself —
those agents expect an image/audio payload, not a string. dispatch() may
still attempt this for OCR/ASR/HYBRID-classified queries (per AGENT_MAP)
and fail harmlessly, since BaseAgent.process() catches the error; we only
ever read the "visual" result out of dispatch()'s return value.
"""

import argparse
import asyncio

import yaml

from .agents.asr_agent import ASRAgent
from .agents.ocr_agent import OCRAgent
from .agents.visual_agent import VisualAgent
from .retrieval.vector_store import VectorStore
from .routing.classifier import QueryType, rule_based_classify
from .routing.dispatcher import DynamicDispatcher

# Process-lifetime cache so repeated search() calls (e.g. from the Part 2
# eval harness, which calls this once per query) don't reload SigLIP /
# Whisper / the FAISS index from scratch every time. Keyed by the config
# dict's identity, which stays stable as long as the caller loads
# config.yaml once and reuses the dict, exactly how eval.py and the CLI
# below both use it.
_CONTEXT_CACHE: dict[int, tuple[dict, VectorStore]] = {}


def _build_agents(config: dict) -> dict:
    agents_cfg = config["agents"]
    visual_cfg = agents_cfg["visual"]
    agents: dict = {
        "visual": VisualAgent(
            model_name=visual_cfg["model"],
            pretrained=visual_cfg["pretrained"],
            max_concurrent=visual_cfg.get("max_concurrent", 8),
        )
    }
    if "asr" in agents_cfg:
        asr_cfg = agents_cfg["asr"]
        agents["asr"] = ASRAgent(
            model_name=asr_cfg["model"],
            language=asr_cfg.get("language", "vi"),
            max_concurrent=asr_cfg.get("max_concurrent", 2),
        )
    if "ocr" in agents_cfg:
        ocr_cfg = agents_cfg["ocr"]
        agents["ocr"] = OCRAgent(
            model_name=ocr_cfg["model"],
            max_concurrent=ocr_cfg.get("max_concurrent", 4),
        )
    return agents


def _get_context(config: dict) -> tuple[dict, VectorStore]:
    key = id(config)
    if key not in _CONTEXT_CACHE:
        agents = _build_agents(config)
        store = VectorStore(
            embed_dim=config["retrieval"]["embed_dim"],
            nlist=config["retrieval"]["nlist"],
            nprobe=config["retrieval"]["nprobe"],
        )
        store.load(config["data"]["embed_dir"])
        _CONTEXT_CACHE[key] = (agents, store)
    return _CONTEXT_CACHE[key]


def _hybrid_rerank(
    query: str, candidates: list[dict], store: VectorStore, visual_weight: float = 0.7
) -> list[dict]:
    """Phase 1 optional step (docs/IMPLEMENTATION_PLAN.md §6): blend visual
    similarity with a BM25 score against stored asr_text/ocr_text for
    queries the classifier flagged as OCR/ASR/HYBRID."""
    from rank_bm25 import BM25Okapi

    docs = []
    for c in candidates:
        fields = store.get_text_fields(c["video_id"], c["frame_idx"])
        text = " ".join(t for t in (fields["asr_text"], fields["ocr_text"]) if t)
        docs.append(text.split() or [""])

    bm25 = BM25Okapi(docs)
    text_scores = bm25.get_scores(query.split())
    max_text = max(text_scores) if max(text_scores) > 0 else 1.0

    reranked = [
        {**c, "score": visual_weight * c["score"] + (1 - visual_weight) * (t / max_text)}
        for c, t in zip(candidates, text_scores)
    ]
    reranked.sort(key=lambda r: r["score"], reverse=True)
    return reranked


async def _search_async(query: str, config: dict, top_k: int) -> list[dict]:
    agents, store = _get_context(config)
    dispatcher = DynamicDispatcher(agents, sla_ms=config["dispatcher"]["sla_latency_ms"])
    clf_output = rule_based_classify(query)

    results = await dispatcher.dispatch({"text": query}, clf_output)
    visual_result = next((r for r in results if r.agent_type == "visual"), None)
    if visual_result is None:
        raise RuntimeError("visual agent not configured — check configs/config.yaml")
    if not visual_result.success:
        raise RuntimeError(f"query embedding failed: {visual_result.error}")

    # Hybrid queries fetch a wider candidate pool so BM25 reranking has
    # something to work with, then get trimmed back to top_k below.
    fetch_k = top_k * 5 if clf_output.query_type != QueryType.TEXT_ONLY else top_k
    candidates = store.search(visual_result.output, top_k=fetch_k)

    if clf_output.query_type != QueryType.TEXT_ONLY and candidates:
        candidates = _hybrid_rerank(query, candidates, store)

    return candidates[:top_k]


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
