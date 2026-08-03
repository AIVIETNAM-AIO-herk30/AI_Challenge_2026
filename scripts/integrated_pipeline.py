"""
Integrated pipeline: video -> keyframe (TransNetV2+DAKE) -> caption
(qwen2.5vl:3b) -> ASR join -> ReCap (qwen2.5:3b-instruct, text-only, grouped).

For each video in --video-dir:
  [1/4] Extract keyframes via scripts/keyframe/transnetv2_dake_keyframes.py's run()
        (skipped if data/keyframe/{video_id}/ already has images)
  [2/4] Caption each keyframe with qwen2.5vl:3b (vision, NOT a thinking model
        like qwen3-vl -- lower num_predict, less retry pressure expected)
  [3/4] Join ASR context per keyframe from data/asr/{video_id}.json's
        raw_segments (fixed +/-window; NOT `intervals`, which is currently
        empty for every video -- group_asr_segments.py was never built)
  [4/4] ReCap: group consecutive keyframes (--recap-group-size), send their
        captions+asr_context+running memory to qwen2.5:3b-instruct
        (text-only) in ONE call per group, producing an enriched recap per
        keyframe + one updated memory for the group. Falls back to
        per-keyframe recap calls if the group response doesn't parse.

Output: data/integrated/{video_id}.json
{
  "video_id": ..., "caption_model": "qwen2.5vl:3b",
  "recap_model": "qwen2.5:3b-instruct",
  "keyframes": [
    {"frame_id":..., "frame_idx":..., "timestamp_sec":..., "shot_index":...,
     "image_path":..., "caption":..., "caption_status": "ok"|"empty"|"truncated",
     "asr_context":..., "recap":..., "_memory_after":...},
    ...
  ]
}
"""
import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.keyframe.transnetv2_dake_keyframes import run as extract_keyframes_run  # noqa: E402

DEFAULT_VIDEO_DIR = "data/test_1video"
DEFAULT_ASR_DIR = "data/asr"
DEFAULT_KEYFRAME_DIR = "data/keyframe"
DEFAULT_OUTPUT_DIR = "data/integrated"
DEFAULT_MODEL_DIR = "weights/transnetv2/"
DEFAULT_CAPTION_MODEL = "qwen2.5vl:3b"
DEFAULT_RECAP_MODEL = "qwen2.5:3b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

CAPTION_PROMPT = (
    "Describe this video keyframe in one concise English sentence. "
    "Focus on the main subject, action, and setting. "
    "Do not mention timestamps, frame numbers, or that this is a video frame."
)

INITIAL_MEMORY = "#Program: unknown\n#Topic: unknown\n#Characters: none identified\n#Notes: none"

RECAP_GROUP_PROMPT_TEMPLATE = """You are analyzing a sequence of {n} consecutive frame captions from a Vietnamese TV news broadcast, in order, to produce temporally consistent, context-enriched descriptions.

Previous memory (context accumulated so far):
{memory}

Frames in order:
{items_block}

For EACH frame, write ONE enriched caption that uses the memory and subtitle to add context (names, topic, prior events) where relevant -- do not invent facts not supported by the caption, subtitle, or memory. Then update the memory for the whole group. Keep the ENTIRE memory block under 100 words.

Respond in EXACTLY this format, nothing else:
RECAP 1: <enriched caption for frame 1>
RECAP 2: <enriched caption for frame 2>
(... one RECAP line per frame ...)
MEMORY:
#Program: <program name or "unknown">
#Topic: <current segment/topic>
#Characters: <list, format "name/description: facts", or "none identified">
#Notes: <any other useful running context, or "none">
"""

RECAP_SINGLE_PROMPT_TEMPLATE = """You are analyzing one frame caption from a Vietnamese TV news broadcast, using running memory to add context.

Previous memory:
{memory}

Caption: {caption}
Subtitle around this moment: {subtitle}

Write ONE enriched caption using the memory/subtitle for context where relevant -- do not invent facts. Then update the memory (keep it under 100 words).

Respond in EXACTLY this format, nothing else:
RECAP: <enriched caption>
MEMORY:
#Program: <program name or "unknown">
#Topic: <current segment/topic>
#Characters: <list, format "name/description: facts", or "none identified">
#Notes: <any other useful running context, or "none">
"""


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def free_ollama(model: str) -> None:
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)


def _call_ollama(prompt: str, model: str, ollama_url: str, num_predict: int,
                  num_ctx: int, images_b64: list[str] | None = None) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"num_predict": num_predict, "temperature": 0.3, "num_ctx": num_ctx},
    }
    if images_b64:
        payload["images"] = images_b64
    response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Step 1: keyframe extraction
# ---------------------------------------------------------------------------

def ensure_keyframes(video_path: Path, video_id: str, keyframe_dir: Path,
                      model_dir: Path, caption_model: str, recap_model: str) -> list[dict]:
    out_dir = keyframe_dir / video_id
    if not (out_dir.exists() and any(out_dir.glob("*.jpg"))):
        print(f"[1/4] {video_id}: freeing Ollama models before CPU-heavy "
              f"TransNetV2+DAKE extraction...")
        free_ollama(caption_model)
        free_ollama(recap_model)
        print(f"[1/4] extracting keyframes for {video_id} (TransNetV2+DAKE)...")
        extract_keyframes_run(
            video_path=video_path, output_dir=keyframe_dir, model_dir=model_dir,
            threshold=0.5, min_shot_duration_sec=2, dake_window=5,
            dake_keyframe_ratio=0.05, dake_min_keyframes_per_shot=1,
            dake_min_gap_sec=0.5, jpeg_quality=90, phash_size=8,
            phash_dup_hamming_threshold=8, color_bins=8,
            color_dup_bhattacharyya_threshold=0.3, apply_overlay_mask=True,
        )
    else:
        print(f"[1/4] {video_id}: keyframes already extracted, skipping")

    records = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    records.sort(key=lambda r: r["frame_idx"])
    print(f"      -> {len(records)} keyframes")
    return records


# ---------------------------------------------------------------------------
# Step 2: captioning (qwen2.5vl:3b)
# ---------------------------------------------------------------------------

def caption_keyframe(image_path: Path, model: str, ollama_url: str, num_predict: int,
                      num_ctx: int, max_retries: int = 2) -> tuple[str, str]:
    """Returns (caption, caption_status). Not a thinking model, so truncation
    should be rarer than qwen3-vl, but still retry with doubled budget."""
    budget = num_predict
    data = _call_ollama(CAPTION_PROMPT, model, ollama_url, budget, num_ctx, [encode_image(image_path)])
    retries = 0
    while data.get("done_reason") == "length" and retries < max_retries:
        budget *= 2
        retries += 1
        data = _call_ollama(CAPTION_PROMPT, model, ollama_url, budget, num_ctx, [encode_image(image_path)])

    text = data["response"].strip()
    if not text:
        return "", "empty"
    if data.get("done_reason") == "length":
        return text, "truncated"
    return text, "ok"


def run_captioning(records: list[dict], model: str, ollama_url: str, num_predict: int,
                    num_ctx: int, restart_every: int, checkpoint_fn, checkpoint_every: int = 10) -> None:
    pending = [r for r in records if r.get("caption_status") != "ok"]
    print(f"[2/4] captioning {len(pending)}/{len(records)} keyframes with {model}...")
    for i, r in enumerate(pending, start=1):
        t0 = time.time()
        caption, status = caption_keyframe(Path(r["image_path"]), model, ollama_url,
                                            num_predict, num_ctx)
        r["caption"], r["caption_status"] = caption, status
        dt = time.time() - t0
        flag = f" [{status}]" if status != "ok" else ""
        print(f"  [{i}/{len(pending)}] {r['frame_id']} ({dt:.1f}s){flag} -> {caption[:100]}")
        if i % checkpoint_every == 0:
            checkpoint_fn()
            print(f"  [checkpoint] wrote progress ({i}/{len(pending)})")
        if restart_every and i % restart_every == 0 and i < len(pending):
            free_ollama(model)
            print(f"  [free-ollama] stopped {model} after {i} images")


# ---------------------------------------------------------------------------
# Step 3: ASR join (fixed +/- window; `intervals` is empty for every video)
# ---------------------------------------------------------------------------

def gather_asr_text(raw_segments: list[dict], center_sec: float, window_sec: float) -> str:
    texts = [seg["text"] for seg in raw_segments
             if center_sec - window_sec <= seg["start"] <= center_sec + window_sec]
    return " ".join(texts).strip()


def run_asr_join(records: list[dict], asr_path: Path, window_sec: float) -> None:
    print(f"[3/4] joining ASR context (+/-{window_sec}s window)...")
    if not asr_path.exists():
        print(f"      [warn] {asr_path} not found -- leaving asr_context empty for all keyframes")
        for r in records:
            r["asr_context"] = ""
        return
    raw_segments = json.loads(asr_path.read_text())["raw_segments"]
    for r in records:
        r["asr_context"] = gather_asr_text(raw_segments, r["timestamp_sec"], window_sec)
    n_with_context = sum(1 for r in records if r["asr_context"])
    print(f"      -> {n_with_context}/{len(records)} keyframes have non-empty asr_context")


# ---------------------------------------------------------------------------
# Step 4: ReCap (text-only, grouped, qwen2.5:3b-instruct)
# ---------------------------------------------------------------------------

RECAP_LINE_RE = re.compile(r"RECAP\s+(\d+):\s*(.+?)(?=RECAP\s+\d+:|MEMORY:|$)", re.DOTALL)
MEMORY_RE = re.compile(r"MEMORY:\s*(.+)", re.DOTALL)
SINGLE_RECAP_RE = re.compile(r"RECAP:\s*(.+?)\s*MEMORY:\s*(.+)", re.DOTALL)


def parse_group_response(text: str, n: int) -> tuple[list[str], str] | None:
    mm = MEMORY_RE.search(text)
    if not mm:
        return None
    memory = mm.group(1).strip()
    recap_section = text[:mm.start()]
    recaps = {}
    for rm in RECAP_LINE_RE.finditer(recap_section):
        recaps[int(rm.group(1))] = rm.group(2).strip()
    if len(recaps) != n or set(recaps) != set(range(1, n + 1)):
        return None
    if not memory or any(not recaps[i] for i in range(1, n + 1)):
        return None
    return [recaps[i] for i in range(1, n + 1)], memory


def recap_single(record: dict, memory: str, model: str, ollama_url: str,
                  num_predict: int, num_ctx: int, max_retries: int = 2) -> tuple[str, str]:
    prompt = RECAP_SINGLE_PROMPT_TEMPLATE.format(
        memory=memory, caption=record.get("caption", ""),
        subtitle=record.get("asr_context") or "(none)")
    budget = num_predict
    data = _call_ollama(prompt, model, ollama_url, budget, num_ctx)
    parsed = SINGLE_RECAP_RE.match(data["response"].strip())
    retries = 0
    while (data.get("done_reason") == "length" or parsed is None) and retries < max_retries:
        budget = min(budget * 2, num_ctx - 500)
        retries += 1
        data = _call_ollama(prompt, model, ollama_url, budget, num_ctx)
        parsed = SINGLE_RECAP_RE.match(data["response"].strip())
    if parsed:
        return parsed.group(1).strip(), parsed.group(2).strip()
    print(f"  [warn] {record['frame_id']}: single-recap fallback also unparsable, "
          f"keeping caption as recap, memory unchanged")
    return record.get("caption", ""), memory


def recap_group(group: list[dict], memory: str, model: str, ollama_url: str,
                 num_predict_per_item: int, num_ctx: int, max_retries: int = 2) -> tuple[list[str], str] | None:
    n = len(group)
    items_block = "\n".join(
        f"Frame {i}:\n  Caption: {r.get('caption', '')}\n"
        f"  Subtitle: {r.get('asr_context') or '(none)'}"
        for i, r in enumerate(group, start=1)
    )
    prompt = RECAP_GROUP_PROMPT_TEMPLATE.format(n=n, memory=memory, items_block=items_block)
    budget = num_predict_per_item * n + 200
    max_budget = num_ctx - 1500
    data = _call_ollama(prompt, model, ollama_url, min(budget, max_budget), num_ctx)
    parsed = parse_group_response(data["response"], n)
    retries = 0
    while (data.get("done_reason") == "length" or parsed is None) and retries < max_retries \
            and budget < max_budget:
        budget = min(budget * 2, max_budget)
        retries += 1
        data = _call_ollama(prompt, model, ollama_url, budget, num_ctx)
        parsed = parse_group_response(data["response"], n)
    return parsed


def run_recap(records: list[dict], group_size: int, model: str, ollama_url: str,
              num_predict_per_item: int, num_ctx: int, checkpoint_fn, restart_every: int) -> None:
    done_ids = {r["frame_id"] for r in records if "recap" in r}
    pending_records = [r for r in records if r["frame_id"] not in done_ids]
    groups = [pending_records[i:i + group_size] for i in range(0, len(pending_records), group_size)]

    memory = INITIAL_MEMORY
    for r in reversed(records):
        if "_memory_after" in r:
            memory = r["_memory_after"]
            break

    print(f"[4/4] recap: {len(pending_records)}/{len(records)} keyframes "
          f"in {len(groups)} groups of <= {group_size}, model={model}")

    n_done = 0
    for gi, group in enumerate(groups, start=1):
        t0 = time.time()
        result = recap_group(group, memory, model, ollama_url, num_predict_per_item, num_ctx)
        if result is not None:
            recaps, memory = result
            for r, recap_text in zip(group, recaps):
                r["recap"], r["_memory_after"] = recap_text, memory
        else:
            print(f"  [warn] group {gi}/{len(groups)} ({len(group)} frames) failed to parse "
                  f"after retries -- falling back to per-frame recap")
            for r in group:
                r["recap"], memory = recap_single(r, memory, model, ollama_url,
                                                   num_predict_per_item, num_ctx)
                r["_memory_after"] = memory
        dt = time.time() - t0
        n_done += len(group)
        print(f"  [{gi}/{len(groups)}] group of {len(group)} frames ({dt:.1f}s) "
              f"-> memory: {memory[:80]}...")

        checkpoint_fn()
        if restart_every and n_done % restart_every < group_size and gi < len(groups):
            free_ollama(model)
            print(f"  [free-ollama] stopped {model} after ~{n_done} keyframes")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path, asr_dir: Path, keyframe_dir: Path, output_dir: Path, model_dir: Path,
    caption_model: str, recap_model: str, ollama_url: str,
    caption_num_predict: int, recap_num_predict_per_item: int, num_ctx: int,
    asr_window_sec: float, recap_group_size: int, restart_every: int, overwrite: bool,
) -> None:
    video_id = video_path.stem
    output_path = output_dir / f"{video_id}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        data = json.loads(output_path.read_text())
        by_frame_id = {r["frame_id"]: r for r in data["keyframes"]}
    else:
        data = {"video_id": video_id, "caption_model": caption_model,
                "recap_model": recap_model, "keyframes": []}
        by_frame_id = {}

    fresh_records = ensure_keyframes(video_path, video_id, keyframe_dir, model_dir,
                                      caption_model, recap_model)
    records = []
    for fr in fresh_records:
        r = by_frame_id.get(fr["frame_id"], fr)
        records.append(r)
    data["keyframes"] = records

    def checkpoint():
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    checkpoint()  # write the keyframe list to disk immediately, before captioning starts

    run_captioning(records, caption_model, ollama_url, caption_num_predict, num_ctx,
                    restart_every, checkpoint)
    checkpoint()

    run_asr_join(records, asr_dir / f"{video_id}.json", asr_window_sec)
    checkpoint()

    run_recap(records, recap_group_size, recap_model, ollama_url,
              recap_num_predict_per_item, num_ctx, checkpoint, restart_every)
    checkpoint()
    print(f"Wrote {output_path}")


def run(video_dir: Path, asr_dir: Path, keyframe_dir: Path, output_dir: Path, model_dir: Path,
        caption_model: str, recap_model: str, ollama_url: str,
        caption_num_predict: int, recap_num_predict_per_item: int, num_ctx: int,
        asr_window_sec: float, recap_group_size: int, restart_every: int, overwrite: bool) -> None:
    video_paths = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(video_paths)} videos in {video_dir}")
    for video_path in video_paths:
        try:
            process_video(video_path, asr_dir, keyframe_dir, output_dir, model_dir,
                           caption_model, recap_model, ollama_url,
                           caption_num_predict, recap_num_predict_per_item, num_ctx,
                           asr_window_sec, recap_group_size, restart_every, overwrite)
        except Exception as e:
            print(f"[LỖI toàn bộ video] {video_path.stem}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--asr-dir", default=DEFAULT_ASR_DIR)
    parser.add_argument("--keyframe-dir", default=DEFAULT_KEYFRAME_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--caption-model", default=DEFAULT_CAPTION_MODEL)
    parser.add_argument("--recap-model", default=DEFAULT_RECAP_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--caption-num-predict", type=int, default=300,
                         help="qwen2.5vl is NOT a thinking model, so this can start much "
                              "lower than qwen3-vl's 400-700 (default: 300)")
    parser.add_argument("--recap-num-predict-per-item", type=int, default=150,
                         help="Per-keyframe token budget within a recap group call "
                              "(total budget = this * group_size + 200 for memory)")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--asr-window-sec", type=float, default=2.0)
    parser.add_argument("--recap-group-size", type=int, default=5)
    parser.add_argument("--restart-every", type=int, default=50,
                         help="Run `ollama stop` every N keyframes to reset server memory "
                              "growth (default: 50, 0 to disable)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run(
        video_dir=Path(args.video_dir), asr_dir=Path(args.asr_dir),
        keyframe_dir=Path(args.keyframe_dir), output_dir=Path(args.output_dir),
        model_dir=Path(args.model_dir), caption_model=args.caption_model,
        recap_model=args.recap_model, ollama_url=args.ollama_url,
        caption_num_predict=args.caption_num_predict,
        recap_num_predict_per_item=args.recap_num_predict_per_item,
        num_ctx=args.num_ctx, asr_window_sec=args.asr_window_sec,
        recap_group_size=args.recap_group_size, restart_every=args.restart_every,
        overwrite=args.overwrite,
    )
