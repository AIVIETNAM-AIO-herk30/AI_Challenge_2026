"""
Caption/recap indexing -- builds the TextualDB vector store.

Reads scripts/integrated_pipeline.py's output (data/integrate/integrated/{video_id}.json),
embeds each keyframe's recap (fallback: caption) with the same SigLIP text
tower used to embed queries at search time (src/inference.py), so caption
vectors land in the same 1152-dim joint space as the image embeddings built
by video_indexer.py. Also upserts caption/recap text into Elasticsearch for
BM25 search (ElasticsearchStore.search_caption).

No Milvus/MobileClip here -- see src/retrieval/vector_store.py's docstring:
this repo's actual vector store is Turbovec, not the paper's Milvus design.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.visual_agent import VisualAgent
from .es_store import ElasticsearchStore
from .vector_store import TurbovecStore


@dataclass
class CaptionIndexReport:
    video_id: str
    n_keyframes: int
    frame_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class CaptionIndexer:
    def __init__(
        self,
        visual_agent: VisualAgent,
        caption_store: TurbovecStore,
        es_store: ElasticsearchStore,
    ):
        self._visual = visual_agent
        self._caption_store = caption_store
        self._es_store = es_store

    async def index_file(self, integrated_path: Path) -> CaptionIndexReport:
        integrated_path = Path(integrated_path)
        data = json.loads(integrated_path.read_text())
        video_id = data.get("video_id", integrated_path.stem)
        keyframes = data.get("keyframes", [])
        report = CaptionIndexReport(video_id=video_id, n_keyframes=len(keyframes))

        es_docs = []
        for kf in keyframes:
            frame_id = kf["frame_id"]
            text = (kf.get("recap") or kf.get("caption") or "").strip()
            if not text:
                report.errors.append(f"{frame_id}: no recap/caption text, skipped")
                continue

            result = await self._visual.process({"text": text})
            if not result.success:
                report.errors.append(f"{frame_id}: embedding failed: {result.error}")
                continue

            self._caption_store.insert(frame_id, result.output)
            es_docs.append(
                {
                    "frame_id": frame_id,
                    "caption": kf.get("caption") or "",
                    "recap": kf.get("recap") or "",
                }
            )
            report.frame_ids.append(frame_id)

        self._es_store.bulk_upsert_captions(es_docs)
        return report

    async def index_directory(self, integrated_dir: str | Path) -> list[CaptionIndexReport]:
        integrated_dir = Path(integrated_dir)
        reports = []
        for integrated_path in sorted(integrated_dir.glob("*.json")):
            reports.append(await self.index_file(integrated_path))
        return reports


async def _build_and_run(config: dict) -> None:
    """CLI entry point wiring: python -m src.retrieval.caption_indexer"""
    agents_cfg = config["agents"]
    turbovec_cfg = config["turbovec"]

    siglip_cfg = agents_cfg["visual"]["siglip"]
    visual_agent = VisualAgent(
        model_name=siglip_cfg["model"],
        pretrained=siglip_cfg["pretrained"],
        max_concurrent=siglip_cfg.get("max_concurrent", 8),
    )
    caption_store = TurbovecStore(dim=siglip_cfg["embed_dim"], bit_width=turbovec_cfg.get("bit_width", 4))

    es_cfg = config["elasticsearch"]
    es_store = ElasticsearchStore(url=es_cfg.get("url"), index_name=es_cfg["index_name"])
    es_store.ensure_index()

    indexer = CaptionIndexer(
        visual_agent=visual_agent,
        caption_store=caption_store,
        es_store=es_store,
    )

    integrated_dir = config.get("data", {}).get("integrated_dir", "data/integrate/integrated/")
    reports = await indexer.index_directory(integrated_dir)
    for r in reports:
        print(f"{r.video_id}: {r.n_keyframes} keyframes, {len(r.frame_ids)} captions indexed, {len(r.errors)} errors")
        for err in r.errors:
            print(f"  ! {err}")

    caption_store.save(Path(turbovec_cfg["index_dir"]) / "caption")

    total_frames = sum(len(r.frame_ids) for r in reports)
    print(f"Done: {total_frames} captions indexed -> {turbovec_cfg['index_dir']}/caption + Elasticsearch")


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    asyncio.run(_build_and_run(cfg))
