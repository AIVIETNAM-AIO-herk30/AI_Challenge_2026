"""
Generate a per-keyframe caption using a local Ollama vision model
(default: qwen3-vl:4b) and merge it into the existing
data/json/{video_id}.json produced by convert_map_keyframes_to_json.py.

Input:
  - data/json/{video_id}.json          (keyframe metadata: n, frame_idx, pts_time, fps, frame_id)
  - data/raw/queries/Keyframes_L21/keyframes/{video_id}/{n:03d}.jpg  (organizer keyframe images)

Output: the same data/json/{video_id}.json, in place, with a "caption"
field added to every keyframe record.

Requires `ollama serve` running locally with the model already pulled:
    ollama pull qwen3-vl:4b
"""
import argparse
import base64
import json
import subprocess
import time
from pathlib import Path

import requests

DEFAULT_JSON_DIR = "data/json"
DEFAULT_KEYFRAME_ROOT = "data/raw/queries/Keyframes_L21/keyframes"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

CAPTION_PROMPT = (
    "Describe this video keyframe in one concise English sentence. "
    "Focus on the main subject, action, and setting. "
    "Do not mention timestamps, frame numbers, or that this is a video frame."
)


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _call_ollama(image_path: Path, model: str, ollama_url: str, num_predict: int, keep_alive: str) -> dict:
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": CAPTION_PROMPT,
            "images": [encode_image(image_path)],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_predict": num_predict, "temperature": 0.3},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def generate_caption(
    image_path: Path,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    max_retries: int = 2,
) -> str:
    # qwen3-vl is a "thinking" model: it always emits internal reasoning
    # before the final answer, and num_predict can run out mid-thought
    # (done_reason == "length"), leaving an empty/truncated response. Retry
    # with doubled budget only when that happens, so the common case stays fast.
    budget = num_predict
    data = _call_ollama(image_path, model, ollama_url, budget, keep_alive)
    retries = 0
    while data.get("done_reason") == "length" and retries < max_retries:
        budget *= 2
        retries += 1
        print(f"  [retry {retries}/{max_retries}] {image_path.name}: truncated at "
              f"num_predict={budget // 2}, retrying with num_predict={budget}")
        data = _call_ollama(image_path, model, ollama_url, budget, keep_alive)

    if data.get("done_reason") == "length":
        print(f"  [warn] {image_path.name}: still truncated after {max_retries} retries "
              f"(num_predict={budget}) -- keeping partial response")
    return data["response"].strip()


def free_ollama(model: str) -> None:
    """Unload the model and kill its underlying llama-server subprocess.

    Ollama's server was observed to leak memory across many sequential
    requests (each /api/generate logs a "prompt_save ... total state size"
    that never shrinks -- see plan notes), eventually triggering a real OOM
    kill after a few hundred images. `ollama stop` forces that subprocess to
    exit, so the next request spawns a fresh one with clean memory -- no
    sudo/systemctl restart needed, just a short reload delay on the next call.
    """
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)


def run(
    video_id: str,
    json_dir: Path,
    keyframe_root: Path,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    limit: int | None,
    checkpoint_every: int,
    overwrite: bool,
    restart_every: int,
) -> None:
    json_path = json_dir / f"{video_id}.json"
    keyframe_dir = keyframe_root / video_id

    data = json.loads(json_path.read_text())
    keyframes = data["keyframes"]
    if limit is not None:
        keyframes = keyframes[:limit]

    # Resume support: a prior run may have crashed (e.g. OOM) partway through.
    # Skip keyframes that already have a caption unless --overwrite is passed.
    pending = [kf for kf in keyframes if overwrite or not kf.get("caption")]
    skipped = len(keyframes) - len(pending)
    if skipped:
        print(f"[caption] resuming: {skipped} keyframes already captioned, skipping")

    total = len(pending)
    print(f"[caption] {video_id}: {total} keyframes to caption, model={model}")

    t_start = time.time()
    for i, kf in enumerate(pending, start=1):
        image_path = keyframe_dir / f"{kf['n']:03d}.jpg"
        t0 = time.time()
        kf["caption"] = generate_caption(image_path, model, ollama_url, num_predict, keep_alive)
        dt = time.time() - t0
        print(f"[{i}/{total}] n={kf['n']:03d} frame_idx={kf['frame_idx']} "
              f"({dt:.1f}s) -> {kf['caption']}")

        # Checkpoint to disk periodically so a crash (e.g. OOM) doesn't lose
        # everything captioned so far -- only the last partial batch is at risk.
        if i % checkpoint_every == 0:
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"  [checkpoint] wrote progress ({i}/{total}) to {json_path}")

        # Ollama's server was observed to leak memory across many sequential
        # requests, eventually OOM-crashing after a few hundred images (see
        # plan notes). Force a fresh subprocess periodically to reset that.
        if restart_every and i % restart_every == 0 and i < total:
            free_ollama(model)
            print(f"  [free-ollama] stopped {model} after {i} images to reset memory "
                  f"(next call will reload it)")

    elapsed = time.time() - t_start
    if total:
        print(f"Done: {total} captions in {elapsed:.1f}s ({elapsed / total:.2f}s/keyframe avg)")
    else:
        print("Done: nothing to caption (all keyframes already had captions)")

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Wrote captions into {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    parser.add_argument("--keyframe-root", default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-predict", type=int, default=400,
                         help="Max tokens to generate per caption (default: 400 -- "
                              "qwen3-vl is a thinking model and needs headroom for its "
                              "internal reasoning before it writes the final answer)")
    parser.add_argument("--keep-alive", default="30m",
                         help="How long Ollama keeps the model loaded between calls (default: 30m)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only caption the first N keyframes (for a quick test run)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                         help="Write progress to disk every N captions (default: 10), "
                              "so a crash mid-run only loses at most one partial batch")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-caption keyframes that already have a caption "
                              "(default: skip them, to resume a crashed run)")
    parser.add_argument("--restart-every", type=int, default=50,
                         help="Run `ollama stop` every N images (default: 50) to reset "
                              "the server's per-request memory growth before it can OOM "
                              "the whole service. Use 0 to disable.")
    args = parser.parse_args()

    run(
        video_id=args.video_id,
        json_dir=Path(args.json_dir),
        keyframe_root=Path(args.keyframe_root),
        model=args.model,
        ollama_url=args.ollama_url,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        overwrite=args.overwrite,
        restart_every=args.restart_every,
    )
