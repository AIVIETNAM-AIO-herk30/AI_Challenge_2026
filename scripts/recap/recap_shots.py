"""
ReCap variant that groups organizer keyframes into real shots (via TransNetV2,
src/retrieval/shot_detector.py) before running the recurrent caption+memory
loop, instead of treating every keyframe as its own step
(scripts/recap_keyframes.py). Written to A/B compare the two groupings on the
same video/time range, per the paper's actual per-shot design (§4.1) vs the
simpler per-keyframe adaptation.

TransNetV2 only supplies shot start/end timestamps from the real .mp4 -- it
does NOT generate any new keyframe images. Organizer keyframes
(data/raw/queries/Keyframes_L21/keyframes/{video_id}/{n:03d}.jpg) are
grouped into shots by their existing pts_time; one representative keyframe
per shot (closest to the shot's midpoint) is sent to the vision LLM, to
avoid the multi-image-per-call reliability issues already found with
qwen3-vl (see scripts/generate_captions_ollama_fast.py's batch experiment).

Output: data/recap_shot/{video_id}.json
{
  "video_id": ..., "model": ...,
  "shots": [
    {"shot_index": 0, "start_sec": 0.0, "end_sec": 3.5,
     "keyframe_ns": [1, 2], "representative_n": 1,
     "asr_context": "...", "caption": "...", "memory_after": "..."},
    ...
  ]
}
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Reuse the tuned prompt/parsing/retry/free-ollama logic as-is -- only the
# grouping stage (shots vs raw keyframes) differs between the two scripts.
from recap_keyframes import (
    INITIAL_MEMORY,
    free_ollama,
    gather_asr_text,
    recap_one,
)

DEFAULT_JSON_DIR = "data/json"
DEFAULT_ASR_DIR = "data/asr"
DEFAULT_KEYFRAME_ROOT = "data/raw/queries/Keyframes_L21/keyframes"
DEFAULT_VIDEO_ROOT = "data/raw/queries/Videos_L21_a/video"
DEFAULT_SHOT_CACHE_DIR = "data/shots"
DEFAULT_RECAP_DIR = "data/recap_shot"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def detect_shots(video_path: Path, cache_dir: Path, max_end_sec: float | None):
    from src.retrieval.shot_detector import ShotDetector

    detector = ShotDetector()
    shots = detector.detect_and_cache(video_path, cache_dir)
    if max_end_sec is not None:
        shots = [s for s in shots if s.start_sec < max_end_sec]
    return shots


def group_keyframes_into_shots(keyframes: list[dict], shots: list) -> list[dict]:
    """Assigns each keyframe (by pts_time) to the shot whose [start_sec,
    end_sec) contains it; keyframes before the first shot or after the last
    are clamped to the nearest one, so no shot is left without an image."""
    groups = []
    for shot in shots:
        in_shot = [kf for kf in keyframes if shot.start_sec <= kf["pts_time"] < shot.end_sec]
        groups.append({"shot": shot, "keyframes": in_shot})

    # Clamp any keyframe outside every shot's range to its nearest shot, so
    # every keyframe (and ideally every shot) has representation.
    covered_ns = {kf["n"] for g in groups for kf in g["keyframes"]}
    orphans = [kf for kf in keyframes if kf["n"] not in covered_ns]
    for kf in orphans:
        nearest = min(groups, key=lambda g: min(
            abs(kf["pts_time"] - g["shot"].start_sec), abs(kf["pts_time"] - g["shot"].end_sec)))
        nearest["keyframes"].append(kf)
        nearest["keyframes"].sort(key=lambda k: k["pts_time"])

    return [g for g in groups if g["keyframes"]]  # drop shots with no organizer keyframe at all


def pick_representative(shot, keyframes: list[dict]) -> dict:
    midpoint = (shot.start_sec + shot.end_sec) / 2
    return min(keyframes, key=lambda kf: abs(kf["pts_time"] - midpoint))


def run(
    video_id: str,
    json_dir: Path,
    asr_dir: Path,
    keyframe_root: Path,
    video_root: Path,
    shot_cache_dir: Path,
    recap_dir: Path,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    num_ctx: int,
    max_end_sec: float | None,
    checkpoint_every: int,
    restart_every: int,
    overwrite: bool,
) -> None:
    keyframe_dir = keyframe_root / video_id
    recap_path = recap_dir / f"{video_id}.json"
    recap_dir.mkdir(parents=True, exist_ok=True)

    keyframes = json.loads((json_dir / f"{video_id}.json").read_text())["keyframes"]
    raw_segments = json.loads((asr_dir / f"{video_id}.json").read_text())["raw_segments"]

    print(f"[recap-shot] detecting shots for {video_id} "
          f"(max_end_sec={max_end_sec})...")
    t_detect = time.time()
    shots = detect_shots(video_root / f"{video_id}.mp4", shot_cache_dir, max_end_sec)
    print(f"[recap-shot] shot detection took {time.time() - t_detect:.1f}s, "
          f"found {len(shots)} shots in range")

    if max_end_sec is not None:
        keyframes = [kf for kf in keyframes if kf["pts_time"] < max_end_sec]
    groups = group_keyframes_into_shots(keyframes, shots)

    if recap_path.exists() and not overwrite:
        data = json.loads(recap_path.read_text())
        done_entries = data["shots"]
    else:
        data = {"video_id": video_id, "model": model, "shots": []}
        done_entries = []

    done_indices = {e["shot_index"] for e in done_entries}
    memory = done_entries[-1]["memory_after"] if done_entries else INITIAL_MEMORY
    pending = [(i, g) for i, g in enumerate(groups) if i not in done_indices]

    print(f"[recap-shot] {video_id}: {len(pending)}/{len(groups)} shots to process, "
          f"model={model}")
    if done_entries:
        print(f"[recap-shot] resuming from shot_index={done_entries[-1]['shot_index']}, "
              f"memory carried forward")

    t_start = time.time()
    for i, (shot_index, group) in enumerate(pending, start=1):
        shot = group["shot"]
        kfs = group["keyframes"]
        rep = pick_representative(shot, kfs)
        image_path = keyframe_dir / f"{rep['n']:03d}.jpg"
        asr_context = gather_asr_text(raw_segments, shot.start_sec, shot.end_sec)

        t0 = time.time()
        caption, memory = recap_one(
            image_path, memory, asr_context, model, ollama_url, num_predict, keep_alive, num_ctx)
        dt = time.time() - t0

        data["shots"].append({
            "shot_index": shot_index,
            "start_sec": shot.start_sec,
            "end_sec": shot.end_sec,
            "keyframe_ns": [kf["n"] for kf in kfs],
            "representative_n": rep["n"],
            "asr_context": asr_context,
            "caption": caption,
            "memory_after": memory,
        })
        print(f"[{i}/{len(pending)}] shot={shot_index} [{shot.start_sec:.1f}-{shot.end_sec:.1f}s] "
              f"rep_n={rep['n']:03d} ({len(kfs)} kf) ({dt:.1f}s) -> {caption}")

        if i % checkpoint_every == 0:
            recap_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"  [checkpoint] wrote progress ({i}/{len(pending)}) to {recap_path}")

        if restart_every and i % restart_every == 0 and i < len(pending):
            free_ollama(model)
            print(f"  [free-ollama] stopped {model} after {i} shots to reset memory")

    elapsed = time.time() - t_start
    if pending:
        print(f"Done: {len(pending)} shots recapped in {elapsed:.1f}s "
              f"({elapsed / len(pending):.2f}s/shot avg)")
    else:
        print("Done: nothing to process (all shots already recapped)")

    recap_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Wrote recap into {recap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    parser.add_argument("--asr-dir", default=DEFAULT_ASR_DIR)
    parser.add_argument("--keyframe-root", default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--shot-cache-dir", default=DEFAULT_SHOT_CACHE_DIR)
    parser.add_argument("--recap-dir", default=DEFAULT_RECAP_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-end-sec", type=float, default=None,
                         help="Only process shots/keyframes starting before this many "
                              "seconds into the video (e.g. 300 for the first 5 minutes)")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--restart-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(
        video_id=args.video_id,
        json_dir=Path(args.json_dir),
        asr_dir=Path(args.asr_dir),
        keyframe_root=Path(args.keyframe_root),
        video_root=Path(args.video_root),
        shot_cache_dir=Path(args.shot_cache_dir),
        recap_dir=Path(args.recap_dir),
        model=args.model,
        ollama_url=args.ollama_url,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        num_ctx=args.num_ctx,
        max_end_sec=args.max_end_sec,
        checkpoint_every=args.checkpoint_every,
        restart_every=args.restart_every,
        overwrite=args.overwrite,
    )
