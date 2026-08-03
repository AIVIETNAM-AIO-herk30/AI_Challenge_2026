"""
Benchmark a local Ollama vision model's per-keyframe captioning speed,
without touching the real data/json/{video_id}.json pipeline output.

Writes one result file per benchmark run to
data/captions/benchmark_{model}_{video_id}_n{limit}.json, so multiple
models (qwen3-vl:4b, moondream, ...) can be compared side by side.
"""
import argparse
import base64
import json
import time
from pathlib import Path

import requests

DEFAULT_KEYFRAME_ROOT = "data/raw/queries/Keyframes_L21/keyframes"
DEFAULT_OUTPUT_DIR = "data/captions"
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


def caption_one(
    image_path: Path,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    max_retries: int = 2,
) -> dict:
    t0 = time.time()
    budget = num_predict
    data = _call_ollama(image_path, model, ollama_url, budget, keep_alive)
    retries = 0
    while data.get("done_reason") == "length" and retries < max_retries:
        budget *= 2
        retries += 1
        data = _call_ollama(image_path, model, ollama_url, budget, keep_alive)
    elapsed = time.time() - t0
    return {
        "elapsed_sec": round(elapsed, 2),
        "caption": data["response"].strip(),
        "truncated": data.get("done_reason") == "length",
        "retries": retries,
        "eval_count": data.get("eval_count"),
    }


def run(
    video_id: str,
    keyframe_root: Path,
    output_dir: Path,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    limit: int,
) -> None:
    keyframe_dir = keyframe_root / video_id
    image_paths = sorted(keyframe_dir.glob("*.jpg"))[:limit]

    print(f"[bench] model={model} video_id={video_id} n={len(image_paths)}")
    results = []
    for i, image_path in enumerate(image_paths, start=1):
        n = int(image_path.stem)
        r = caption_one(image_path, model, ollama_url, num_predict, keep_alive)
        r["n"] = n
        results.append(r)
        flag = f" [TRUNCATED after {r['retries']} retries]" if r["truncated"] else (
            f" [retried {r['retries']}x]" if r["retries"] else "")
        print(f"[{i}/{len(image_paths)}] n={n:03d} ({r['elapsed_sec']}s){flag} -> {r['caption']}")

    times = [r["elapsed_sec"] for r in results]
    summary = {
        "model": model,
        "video_id": video_id,
        "limit": limit,
        "num_predict": num_predict,
        "total_sec": round(sum(times), 2),
        "avg_sec": round(sum(times) / len(times), 2),
        "min_sec": min(times),
        "max_sec": max(times),
        "num_truncated": sum(1 for r in results if r["truncated"]),
    }
    print(f"Summary: {summary}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_safe = model.replace(":", "-").replace("/", "-")
    output_path = output_dir / f"benchmark_{model_safe}_{video_id}_n{limit}.json"
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--keyframe-root", default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="qwen3-vl:4b")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-predict", type=int, default=400)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    run(
        video_id=args.video_id,
        keyframe_root=Path(args.keyframe_root),
        output_dir=Path(args.output_dir),
        model=args.model,
        ollama_url=args.ollama_url,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        limit=args.limit,
    )
