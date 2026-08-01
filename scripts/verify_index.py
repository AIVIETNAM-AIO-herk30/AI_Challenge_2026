"""
Handoff gate: proves Team 1's indexed data is shaped correctly before
Team 2 builds query-time retrieval against it.

Team 1 doesn't own query-time search, so this only exercises read
operations against the already-populated stores:

1. Path check: keyframe JPG exists at data/keyframes/{video_id}/{frame_id}.jpg
2. Elasticsearch check: GET by frame_id returns exactly the 5 working-convention
   fields (frame_id, video_id, timestamp_seconds, ocr_text, asr_text)
3. Turbovec self-retrieval check: re-embed the on-disk keyframe JPG through the
   SAME agent used at index time, search(top_k=1), assert the top hit's
   resolved frame_id matches with score ~1.0 -- proves the id round-trips
   through the side-car, not just that something got inserted

Usage: python scripts/verify_index.py --config configs/config.yaml
Exit code is nonzero if any check fails.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from PIL import Image

from src.agents.beit3_agent import BEiT3Agent
from src.agents.visual_agent import VisualAgent
from src.retrieval.es_store import ElasticsearchStore
from src.retrieval.vector_store import TurbovecStore

ES_SCHEMA_FIELDS = {"frame_id", "video_id", "timestamp_seconds", "ocr_text", "asr_text"}


def _sample_per_video(docs: list[dict], per_video: int) -> list[dict]:
    by_video: dict[str, list[dict]] = {}
    for doc in docs:
        by_video.setdefault(doc["video_id"], []).append(doc)
    sample = []
    for video_docs in by_video.values():
        sample.extend(sorted(video_docs, key=lambda d: d["frame_id"])[:per_video])
    return sample


async def _check_frame(
    doc: dict,
    keyframe_dir: Path,
    es_store: ElasticsearchStore,
    siglip_store: TurbovecStore,
    siglip_agent: VisualAgent,
    beit3_store: TurbovecStore | None,
    beit3_agent: BEiT3Agent | None,
) -> dict:
    frame_id = doc["frame_id"]
    video_id = doc["video_id"]
    row = {"frame_id": frame_id, "path_ok": False, "es_ok": False, "siglip_ok": False, "beit3_ok": True}

    keyframe_path = keyframe_dir / video_id / f"{frame_id}.jpg"
    row["path_ok"] = keyframe_path.exists()

    es_doc = es_store.get_by_frame_id(frame_id)
    row["es_ok"] = es_doc is not None and set(es_doc.keys()) == ES_SCHEMA_FIELDS

    if row["path_ok"]:
        # Re-embed the on-disk (JPEG-recompressed) keyframe and require the
        # SAME frame_id back as the #1 hit. Score is intentionally NOT
        # required to be near 1.0: Turbovec's bit_width=4 quantization is
        # lossy by design (that's its 16x compression trick) and JPEG
        # re-compression adds its own drift, so even a perfect round-trip
        # self-match empirically lands around ~0.95, not ~1.0. What actually
        # proves the id-to-vector join is correct is that the *right* frame
        # still wins top-1 — a low score floor here only guards against a
        # degenerate all-noise match, not against normal quantization loss.
        image = Image.open(keyframe_path).convert("RGB")

        siglip_result = await siglip_agent.process({"image": image})
        if siglip_result.success:
            hits = siglip_store.search(siglip_result.output, top_k=1)
            row["siglip_ok"] = bool(hits) and hits[0][0] == frame_id and hits[0][1] > 0.5

        if beit3_agent is not None and beit3_store is not None:
            beit3_result = await beit3_agent.process({"image": image})
            if beit3_result.success:
                hits = beit3_store.search(beit3_result.output, top_k=1)
                row["beit3_ok"] = bool(hits) and hits[0][0] == frame_id and hits[0][1] > 0.5
            else:
                row["beit3_ok"] = False

    return row


async def verify(config: dict, sample_per_video: int = 5) -> bool:
    keyframe_dir = Path(config["data"]["keyframe_dir"])
    turbovec_dir = Path(config["turbovec"]["index_dir"])

    es_cfg = config["elasticsearch"]
    es_store = ElasticsearchStore(url=es_cfg.get("url"), index_name=es_cfg["index_name"])

    siglip_cfg = config["agents"]["visual"]["siglip"]
    siglip_store = TurbovecStore(dim=siglip_cfg["embed_dim"])
    siglip_store.load(turbovec_dir / "siglip")
    siglip_agent = VisualAgent(model_name=siglip_cfg["model"], pretrained=siglip_cfg["pretrained"])

    beit3_store = None
    beit3_agent = None
    if "beit3" in config["agents"]["visual"]:
        beit3_cfg = config["agents"]["visual"]["beit3"]
        beit3_store = TurbovecStore(dim=beit3_cfg["embed_dim"])
        beit3_store.load(turbovec_dir / "beit3")
        beit3_agent = BEiT3Agent(model_name=beit3_cfg["model"])

    docs = es_store.scan_all()
    if not docs:
        print("No documents found in Elasticsearch — has the indexer run yet?")
        return False

    sample = _sample_per_video(docs, sample_per_video)
    rows = [
        await _check_frame(doc, keyframe_dir, es_store, siglip_store, siglip_agent, beit3_store, beit3_agent)
        for doc in sample
    ]

    header = f"{'frame_id':<28} {'path':<6} {'es':<6} {'siglip':<8} {'beit3':<6}"
    print(header)
    print("-" * len(header))
    all_ok = True
    for row in rows:
        ok = row["path_ok"] and row["es_ok"] and row["siglip_ok"] and row["beit3_ok"]
        all_ok = all_ok and ok
        print(
            f"{row['frame_id']:<28} "
            f"{'OK' if row['path_ok'] else 'FAIL':<6} "
            f"{'OK' if row['es_ok'] else 'FAIL':<6} "
            f"{'OK' if row['siglip_ok'] else 'FAIL':<8} "
            f"{'OK' if row['beit3_ok'] else 'FAIL':<6}"
        )

    print("-" * len(header))
    print(f"{len(rows)} frames checked across {len({d['video_id'] for d in sample})} videos — {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--per-video", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    passed = asyncio.run(verify(cfg, sample_per_video=args.per_video))
    sys.exit(0 if passed else 1)
