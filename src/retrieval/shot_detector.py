"""
TransNetV2 shot-boundary keyframing.

Standalone stage, deliberately kept out of the same process as the PyTorch
agents (SigLIP/BEiT-3/Whisper): TensorFlow grabs the whole GPU by default,
and on an 8GB card that fights the PyTorch stack for VRAM. TransNetV2 is a
tiny model that gains nothing from the GPU anyway, so it's forced onto CPU
here rather than negotiated for shared GPU access.

Note on weights: the `transnetv2` PyPI package (installed via its git
dependency) ships its bundled weights as unfetched Git LFS pointer files
(verified: `pip`/`uv`'s plain git clone does not run `git lfs pull`) — the
default `TransNetV2()` constructor will fail against the installed package.
Fetch the real weights once (see scripts/fetch_transnetv2_weights.sh) into
`weights/transnetv2/` and pass that path explicitly via `model_dir`,
matching `configs/config.yaml`'s `transnetv2.model_dir`.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Shot:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


class ShotDetector:
    def __init__(
        self,
        model_dir: str | Path = "weights/transnetv2/",
        threshold: float = 0.5,
        min_shot_duration_sec: float = 0.5,
    ):
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        from transnetv2 import TransNetV2

        self._model = TransNetV2(model_dir=str(model_dir))
        self.threshold = threshold
        self.min_shot_duration_sec = min_shot_duration_sec

    def detect(self, video_path: str | Path) -> list[Shot]:
        native_fps = self._get_fps(video_path)
        _frames, single_frame_pred, _all_frame_pred = self._model.predict_video(str(video_path))
        scenes = self._model.predictions_to_scenes(single_frame_pred, threshold=self.threshold)

        shots = []
        for start_frame, end_frame in scenes:
            start_frame, end_frame = int(start_frame), int(end_frame)
            start_sec = start_frame / native_fps
            end_sec = end_frame / native_fps
            if end_sec - start_sec < self.min_shot_duration_sec:
                continue
            shots.append(Shot(start_frame, end_frame, start_sec, end_sec))
        return shots

    def detect_and_cache(self, video_path: str | Path, cache_dir: str | Path) -> list[Shot]:
        video_path = Path(video_path)
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f"{video_path.stem}.json"

        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            return [Shot(**s) for s in payload]

        shots = self.detect(video_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps([asdict(s) for s in shots]))
        return shots

    @staticmethod
    def _get_fps(video_path: str | Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if not fps or np.isnan(fps) or fps <= 0:
            raise ValueError(f"Could not read a valid FPS for {video_path}")
        return fps


def _run_cli(config: dict) -> None:
    """CLI entry point: python -m src.retrieval.shot_detector"""
    tnv2_cfg = config["transnetv2"]
    detector = ShotDetector(
        model_dir=tnv2_cfg.get("model_dir", "weights/transnetv2/"),
        threshold=tnv2_cfg.get("threshold", 0.5),
        min_shot_duration_sec=tnv2_cfg.get("min_shot_duration_sec", 0.5),
    )
    video_dir = Path(config["data"]["video_dir"])
    cache_dir = Path(config["data"]["shots_cache_dir"])

    for video_path in sorted(video_dir.glob("*.mp4")):
        shots = detector.detect_and_cache(video_path, cache_dir)
        print(f"{video_path.stem}: {len(shots)} shots -> {cache_dir / (video_path.stem + '.json')}")


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _run_cli(cfg)
