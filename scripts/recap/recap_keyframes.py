"""
ReCap: temporally-consistent per-keyframe captioning (paper: U-CESE §4.1
"Recurrence Captioning"), adapted to run per-keyframe -- no 
 detection,
no re-scanning the video.

For each organizer keyframe (in order of n), the vision LLM (qwen3-vl:4b) is
given:
  - the keyframe image itself
    (data/raw/queries/Keyframes_L21/keyframes/{video_id}/{n:03d}.jpg)
  - the subtitle text spoken around that keyframe (windowed from
    data/asr/{video_id}.json's raw_segments, between the midpoint of the
    previous keyframe's timestamp and the midpoint of the next one)
  - a running memory string (structured tags: #Program/#Topic/#Characters/#Notes)
and generates BOTH a caption for this keyframe AND an updated memory string,
carrying context forward recurrently -- this is the mechanism the paper
actually uses (the LVLM looks at the image at every step), not a text-only
enrichment pass over pre-generated captions.

Input:
  - data/json/{video_id}.json           (keyframe n/frame_idx/pts_time)
  - data/asr/{video_id}.json             (raw_segments: start/end/text)
  - data/raw/queries/Keyframes_L21/keyframes/{video_id}/{n:03d}.jpg

Output: data/recap/{video_id}.json
{
  "video_id": "L21_V001",
  "model": "qwen3-vl:4b",
  "keyframes": [
    {"n": 1, "frame_idx": 0, "pts_time": 0.0, "asr_context": "...",
     "caption": "...", "memory_after": "#Program: ...\\n#Topic: ...\\n..."},
    ...
  ]
}

Resume note: each step's prompt depends on the PREVIOUS step's memory, so
resuming replays from the last successfully-written keyframe's memory_after
rather than independently skipping completed keyframes.

Requires `ollama serve` running locally with the model already pulled:
    ollama pull qwen3-vl:4b
"""
import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path

import requests

DEFAULT_JSON_DIR = "data/json"
DEFAULT_ASR_DIR = "data/asr"
DEFAULT_KEYFRAME_ROOT = "data/raw/queries/Keyframes_L21/keyframes"
DEFAULT_RECAP_DIR = "data/recap"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

INITIAL_MEMORY = "#Program: unknown\n#Topic: unknown\n#Characters: none identified\n#Notes: none"

RECAP_PROMPT_TEMPLATE = """You are analyzing a Vietnamese TV news broadcast frame-by-frame, maintaining a running memory of characters, topics, and events as you go, in order to produce temporally consistent captions.

Previous memory (context accumulated so far):
{memory}

Subtitle spoken around this moment (may be empty):
{subtitle}

Look at the image (the current keyframe) and:
1. Write ONE concise English caption describing what's visible in THIS frame, using the subtitle and memory to add context (names, topic, prior events) where relevant -- but do not invent facts not supported by the image, subtitle, or memory.
2. Update the memory: keep relevant ongoing context (recurring characters, topic, program segment), drop anything no longer relevant, add new facts introduced by this subtitle/frame. Keep the ENTIRE memory block under 100 words total -- summarize and drop stale details rather than letting it grow indefinitely.

Respond in EXACTLY this format, nothing else:
CAPTION: <one paragraph>
MEMORY:
#Program: <program name or "unknown">
#Topic: <current segment/topic>
#Characters: <list, format "name/description: facts", or "none identified">
#Notes: <any other useful running context, or "none">
"""

RESPONSE_RE = re.compile(r"CAPTION:\s*(.+?)\s*MEMORY:\s*(.+)", re.DOTALL)


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def build_asr_windows(keyframes: list[dict]) -> list[tuple[float, float]]:
    """Non-overlapping windows per keyframe, split at the midpoint between
    each keyframe and its neighbors -- covers every ASR segment exactly once."""
    times = [kf["pts_time"] for kf in keyframes]
    windows = []
    for i in range(len(times)):
        start = (times[i - 1] + times[i]) / 2 if i > 0 else 0.0
        end = (times[i] + times[i + 1]) / 2 if i < len(times) - 1 else float("inf")
        windows.append((start, end))
    return windows


def gather_asr_text(raw_segments: list[dict], start: float, end: float) -> str:
    texts = [seg["text"] for seg in raw_segments if start <= seg["start"] < end]
    return " ".join(texts).strip()


def parse_recap_response(text: str) -> tuple[str, str] | None:
    m = RESPONSE_RE.match(text.strip())
    if not m:
        return None
    caption = m.group(1).strip()
    memory = m.group(2).strip()
    if not caption or not memory:
        return None
    return caption, memory


def free_ollama(model: str) -> None:
    """See scripts/generate_captions_ollama.py for why: Ollama's server was
    observed to leak memory across many sequential requests, eventually
    OOM-killing the whole service. `ollama stop` forces a fresh subprocess."""
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)


def _call_ollama(image_path: Path, prompt: str, model: str, ollama_url: str,
                  num_predict: int, keep_alive: str, num_ctx: int) -> dict:
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [encode_image(image_path)],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_predict": num_predict, "temperature": 0.3, "num_ctx": num_ctx},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def recap_one(
    image_path: Path,
    memory: str,
    subtitle: str,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    num_ctx: int,
    max_retries: int = 2,
) -> tuple[str, str]:
    """Returns (caption, memory_after). Retries with doubled num_predict on
    truncation or unparsable response; falls back to keeping memory
    unchanged + a plain no-context caption if all retries fail, rather than
    losing the whole run over one bad frame.

    The retry budget is capped well below num_ctx: the server rejects a
    request outright (HTTP 400) if prompt tokens + num_predict exceeds the
    loaded context size, rather than truncating gracefully like a too-low
    num_predict does -- doubling forever without a ceiling will eventually
    hit that wall."""
    prompt = RECAP_PROMPT_TEMPLATE.format(memory=memory, subtitle=subtitle or "(none)")
    max_budget = num_ctx - 2500  # leave headroom for image tokens + prompt text
    budget = num_predict
    data = _call_ollama(image_path, prompt, model, ollama_url, budget, keep_alive, num_ctx)
    parsed = parse_recap_response(data["response"])
    retries = 0
    while (data.get("done_reason") == "length" or parsed is None) and retries < max_retries \
            and budget < max_budget:
        budget = min(budget * 2, max_budget)
        retries += 1
        print(f"  [retry {retries}/{max_retries}] {image_path.name}: "
              f"{'truncated' if data.get('done_reason') == 'length' else 'unparsable'}, "
              f"retrying with num_predict={budget}")
        data = _call_ollama(image_path, prompt, model, ollama_url, budget, keep_alive, num_ctx)
        parsed = parse_recap_response(data["response"])

    if parsed is not None:
        return parsed

    print(f"  [warn] {image_path.name}: could not get a parsable CAPTION/MEMORY response "
          f"after {retries} retries -- falling back to a plain caption, memory unchanged")
    fallback_prompt = (
        "Describe this video keyframe in one concise English sentence. "
        "Focus on the main subject, action, and setting."
    )
    fallback = _call_ollama(image_path, fallback_prompt, model, ollama_url, num_predict,
                             keep_alive, num_ctx)
    return fallback["response"].strip(), memory


def run(
    video_id: str,
    json_dir: Path,
    asr_dir: Path,
    keyframe_root: Path,
    recap_dir: Path,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    num_ctx: int,
    limit: int | None,
    checkpoint_every: int,
    restart_every: int,
    overwrite: bool,
) -> None:
    keyframe_dir = keyframe_root / video_id
    recap_path = recap_dir / f"{video_id}.json"
    recap_dir.mkdir(parents=True, exist_ok=True)

    keyframes = json.loads((json_dir / f"{video_id}.json").read_text())["keyframes"]
    if limit is not None:
        keyframes = keyframes[:limit]
    raw_segments = json.loads((asr_dir / f"{video_id}.json").read_text())["raw_segments"]
    windows = build_asr_windows(keyframes)

    if recap_path.exists() and not overwrite:
        data = json.loads(recap_path.read_text())
        done_entries = data["keyframes"]
    else:
        data = {"video_id": video_id, "model": model, "keyframes": []}
        done_entries = []

    done_ns = {e["n"] for e in done_entries}
    memory = done_entries[-1]["memory_after"] if done_entries else INITIAL_MEMORY
    pending = [(kf, w) for kf, w in zip(keyframes, windows) if kf["n"] not in done_ns]

    total = len(keyframes)
    print(f"[recap] {video_id}: {len(pending)}/{total} keyframes to process, model={model}")
    if done_entries:
        print(f"[recap] resuming from n={done_entries[-1]['n']}, "
              f"memory carried forward from that step")

    t_start = time.time()
    for i, (kf, (w_start, w_end)) in enumerate(pending, start=1):
        image_path = keyframe_dir / f"{kf['n']:03d}.jpg"
        asr_context = gather_asr_text(raw_segments, w_start, w_end)

        t0 = time.time()
        caption, memory = recap_one(
            image_path, memory, asr_context, model, ollama_url, num_predict, keep_alive, num_ctx)
        dt = time.time() - t0

        data["keyframes"].append({
            "n": kf["n"],
            "frame_idx": kf["frame_idx"],
            "pts_time": kf["pts_time"],
            "asr_context": asr_context,
            "caption": caption,
            "memory_after": memory,
        })
        print(f"[{i}/{len(pending)}] n={kf['n']:03d} frame_idx={kf['frame_idx']} "
              f"({dt:.1f}s) -> {caption}")

        if i % checkpoint_every == 0:
            recap_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"  [checkpoint] wrote progress ({i}/{len(pending)}) to {recap_path}")

        if restart_every and i % restart_every == 0 and i < len(pending):
            free_ollama(model)
            print(f"  [free-ollama] stopped {model} after {i} images to reset memory "
                  f"(next call will reload it)")

    elapsed = time.time() - t_start
    if pending:
        print(f"Done: {len(pending)} keyframes recapped in {elapsed:.1f}s "
              f"({elapsed / len(pending):.2f}s/keyframe avg)")
    else:
        print("Done: nothing to process (all keyframes already recapped)")

    recap_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Wrote recap into {recap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="L21_V001")
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    parser.add_argument("--asr-dir", default=DEFAULT_ASR_DIR)
    parser.add_argument("--keyframe-root", default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--recap-dir", default=DEFAULT_RECAP_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-predict", type=int, default=700,
                         help="Base num_predict (default: 700 -- higher than plain "
                              "captioning's 400 because the response must contain both "
                              "a caption AND a structured memory block; not yet measured "
                              "against real timing, tune after benchmarking)")
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--num-ctx", type=int, default=8192,
                         help="Context window loaded for the model (default: 8192, up from "
                              "Ollama's auto-selected 4096 -- the recurrent memory + subtitle "
                              "+ image tokens can otherwise exceed the smaller default and the "
                              "server hard-rejects the request instead of truncating gracefully)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N keyframes (for a quick test run)")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--restart-every", type=int, default=50,
                         help="Run `ollama stop` every N keyframes (default: 50) to reset "
                              "server memory growth. Use 0 to disable.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Start over from scratch instead of resuming from an "
                              "existing data/recap/{video_id}.json")
    args = parser.parse_args()

    run(
        video_id=args.video_id,
        json_dir=Path(args.json_dir),
        asr_dir=Path(args.asr_dir),
        keyframe_root=Path(args.keyframe_root),
        recap_dir=Path(args.recap_dir),
        model=args.model,
        ollama_url=args.ollama_url,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        num_ctx=args.num_ctx,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
        restart_every=args.restart_every,
        overwrite=args.overwrite,
    )
