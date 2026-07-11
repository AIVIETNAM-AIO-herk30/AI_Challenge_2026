"""
Video indexing orchestrator — Team 1's offline pipeline.
Owner: Team 1

Wires together: TransNetV2 shot detection -> dual vision-encoder embedding
(SigLIP + BEiT-3) -> Turbovec -> Whisper ASR + Gemini OCR -> Elasticsearch,
producing the working conventions from docs/API_CONTRACT.md:
  frame_id      = "{video_id}_{frame_index:06d}"
  keyframe path = data/keyframes/{video_id}/{frame_id}.jpg
  ES doc        = {frame_id, video_id, timestamp_seconds, ocr_text, asr_text}

Frame decode is sequential (cv2.VideoCapture.read() in a single forward
pass), not index-seek (cv2.CAP_PROP_POS_FRAMES) — seeking isn't reliably
frame-accurate across all H.264 GOP structures, and at 5-10 short sample
videos a full sequential decode costs nothing worth optimizing for yet.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image

from ..agents.asr_agent import ASRAgent
from ..agents.beit3_agent import BEiT3Agent
from ..agents.ocr_agent import OCRAgent
from ..agents.visual_agent import VisualAgent
from .es_store import ElasticsearchStore
from .shot_detector import ShotDetector
from .vector_store import TurbovecStore


@dataclass
class IndexReport:
    video_id: str
    n_shots: int
    frame_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _join_asr_to_shot(segments: list[dict], shot_start_sec: float, shot_end_sec: float) -> str:
    """Overlap join: any ASR segment overlapping [shot_start_sec, shot_end_sec]
    contributes its text, concatenated in order. "" (never None) if nothing
    overlaps -- expected on ambient-noise-only shots, and required by the
    Elasticsearch schema's asr_text: str typing."""
    overlapping = [
        seg["text"].strip()
        for seg in segments
        if seg["start"] < shot_end_sec and seg["end"] > shot_start_sec
    ]
    return " ".join(t for t in overlapping if t)


def _grab_frames(video_path: Path, frame_indices: set[int]) -> dict[int, Image.Image]:
    """Single sequential decode pass, not per-frame seeking -- see module docstring."""
    cap = cv2.VideoCapture(str(video_path))
    remaining = set(frame_indices)
    frames: dict[int, Image.Image] = {}
    idx = 0
    try:
        while remaining:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx in remaining:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames[idx] = Image.fromarray(frame_rgb)
                remaining.discard(idx)
            idx += 1
    finally:
        cap.release()
    return frames


class VideoIndexer:
    def __init__(
        self,
        shot_detector: ShotDetector,
        siglip_agent: VisualAgent,
        siglip_store: TurbovecStore,
        es_store: ElasticsearchStore,
        beit3_agent: BEiT3Agent | None = None,
        beit3_store: TurbovecStore | None = None,
        asr_agent: ASRAgent | None = None,
        ocr_agent: OCRAgent | None = None,
        keyframe_dir: str | Path = "data/keyframes",
        shots_cache_dir: str | Path = "data/interim/shots",
    ):
        self._shot_detector = shot_detector
        self._siglip = siglip_agent
        self._beit3 = beit3_agent
        self._asr = asr_agent
        self._ocr = ocr_agent
        self._siglip_store = siglip_store
        self._beit3_store = beit3_store
        self._es_store = es_store
        self._keyframe_dir = Path(keyframe_dir)
        self._shots_cache_dir = Path(shots_cache_dir)

    async def index_video(self, video_path: Path) -> IndexReport:
        video_path = Path(video_path)
        video_id = video_path.stem
        report = IndexReport(video_id=video_id, n_shots=0)

        shots = self._shot_detector.detect_and_cache(video_path, self._shots_cache_dir)
        report.n_shots = len(shots)
        if not shots:
            return report

        native_fps = ShotDetector._get_fps(video_path)
        asr_segments = await self._transcribe(video_path)

        mid_frames = {(s.start_frame + s.end_frame) // 2 for s in shots}
        frames = _grab_frames(video_path, mid_frames)

        out_dir = self._keyframe_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        es_docs = []
        for shot in shots:
            frame_index = (shot.start_frame + shot.end_frame) // 2
            image = frames.get(frame_index)
            if image is None:
                report.errors.append(f"frame {frame_index} could not be decoded")
                continue

            frame_id = f"{video_id}_{frame_index:06d}"
            timestamp_seconds = frame_index / native_fps
            keyframe_path = out_dir / f"{frame_id}.jpg"
            image.save(keyframe_path, quality=90)

            siglip_result = await self._siglip.process({"image": image})
            if siglip_result.success:
                self._siglip_store.insert(frame_id, siglip_result.output)
            else:
                report.errors.append(f"{frame_id}: siglip failed: {siglip_result.error}")

            if self._beit3 is not None and self._beit3_store is not None:
                beit3_result = await self._beit3.process({"image": image})
                if beit3_result.success:
                    self._beit3_store.insert(frame_id, beit3_result.output)
                else:
                    report.errors.append(f"{frame_id}: beit3 failed: {beit3_result.error}")

            ocr_text = await self._extract_text(keyframe_path)
            asr_text = _join_asr_to_shot(asr_segments, shot.start_sec, shot.end_sec)

            es_docs.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "timestamp_seconds": timestamp_seconds,
                    "ocr_text": ocr_text,
                    "asr_text": asr_text,
                }
            )
            report.frame_ids.append(frame_id)

        self._es_store.bulk_upsert(es_docs)
        return report

    async def index_directory(self, video_dir: str | Path) -> list[IndexReport]:
        video_dir = Path(video_dir)
        reports = []
        for video_path in sorted(video_dir.glob("*.mp4")):
            reports.append(await self.index_video(video_path))
        return reports

    async def _transcribe(self, video_path: Path) -> list[dict]:
        if self._asr is None:
            return []
        result = await self._asr.process(video_path)
        return result.output["segments"] if result.success else []

    async def _extract_text(self, keyframe_path: Path) -> str:
        if self._ocr is None:
            return ""
        result = await self._ocr.process(keyframe_path)
        return result.output if result.success else ""


async def _build_and_run(config: dict) -> None:
    """CLI entry point wiring: python -m src.retrieval.video_indexer"""
    tnv2_cfg = config["transnetv2"]
    agents_cfg = config["agents"]
    turbovec_cfg = config["turbovec"]

    shot_detector = ShotDetector(
        model_dir=tnv2_cfg.get("model_dir", "weights/transnetv2/"),
        threshold=tnv2_cfg.get("threshold", 0.5),
        min_shot_duration_sec=tnv2_cfg.get("min_shot_duration_sec", 0.5),
    )

    siglip_cfg = agents_cfg["visual"]["siglip"]
    siglip_agent = VisualAgent(
        model_name=siglip_cfg["model"],
        pretrained=siglip_cfg["pretrained"],
        max_concurrent=siglip_cfg.get("max_concurrent", 8),
    )
    siglip_store = TurbovecStore(dim=siglip_cfg["embed_dim"], bit_width=turbovec_cfg.get("bit_width", 4))

    beit3_agent = None
    beit3_store = None
    if "beit3" in agents_cfg["visual"]:
        beit3_cfg = agents_cfg["visual"]["beit3"]
        beit3_agent = BEiT3Agent(
            model_name=beit3_cfg["model"], max_concurrent=beit3_cfg.get("max_concurrent", 4)
        )
        beit3_store = TurbovecStore(dim=beit3_cfg["embed_dim"], bit_width=turbovec_cfg.get("bit_width", 4))

    asr_agent = (
        ASRAgent(
            model_name=agents_cfg["asr"]["model"],
            language=agents_cfg["asr"].get("language", "vi"),
            max_concurrent=agents_cfg["asr"].get("max_concurrent", 2),
        )
        if "asr" in agents_cfg
        else None
    )
    ocr_agent = (
        OCRAgent(
            model_name=agents_cfg["ocr"]["model"],
            max_concurrent=agents_cfg["ocr"].get("max_concurrent", 4),
        )
        if "ocr" in agents_cfg
        else None
    )

    es_cfg = config["elasticsearch"]
    es_store = ElasticsearchStore(url=es_cfg.get("url"), index_name=es_cfg["index_name"])
    es_store.ensure_index()

    indexer = VideoIndexer(
        shot_detector=shot_detector,
        siglip_agent=siglip_agent,
        siglip_store=siglip_store,
        es_store=es_store,
        beit3_agent=beit3_agent,
        beit3_store=beit3_store,
        asr_agent=asr_agent,
        ocr_agent=ocr_agent,
        keyframe_dir=config["data"]["keyframe_dir"],
        shots_cache_dir=config["data"]["shots_cache_dir"],
    )

    reports = await indexer.index_directory(config["data"]["video_dir"])
    for r in reports:
        print(f"{r.video_id}: {r.n_shots} shots, {len(r.frame_ids)} frames indexed, {len(r.errors)} errors")
        for err in r.errors:
            print(f"  ! {err}")

    siglip_store.save(Path(turbovec_cfg["index_dir"]) / "siglip")
    if beit3_store is not None:
        beit3_store.save(Path(turbovec_cfg["index_dir"]) / "beit3")

    total_frames = sum(len(r.frame_ids) for r in reports)
    print(f"Done: {total_frames} frames indexed -> {turbovec_cfg['index_dir']} + Elasticsearch")


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    asyncio.run(_build_and_run(cfg))
