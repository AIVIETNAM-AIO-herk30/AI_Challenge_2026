"""
Faster variant of generate_captions_ollama.py: resizes keyframes before sending
them to Ollama, skips near-duplicate frames via perceptual hash (reusing near
neighbors' captions instead of re-querying the model), and can batch several
images into a single /api/generate call to amortize the "thinking" overhead of
qwen3-vl across multiple frames.

Does NOT touch generate_captions_ollama.py or its output file. Defaults to
video_id=L21_V002 so it can be exercised/benchmarked without racing the real
full-run job that may still be captioning L21_V001.

Requires `ollama serve` running locally with the model already pulled:
    ollama pull qwen3-vl:4b
"""
import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import imagehash
import requests
from PIL import Image

DEFAULT_JSON_DIR = "data/json"
DEFAULT_KEYFRAME_ROOT = "data/raw/queries/Keyframes_L21/keyframes"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VIDEO_ID = "L21_V002"

SINGLE_PROMPT = (
    "Describe this video keyframe in one concise English sentence. "
    "Focus on the main subject, action, and setting. "
    "Do not mention timestamps, frame numbers, or that this is a video frame."
)

BATCH_LINE_RE = re.compile(r"^\s*Image\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)


def build_batch_prompt(n: int) -> str:
    return (
        f"You are shown {n} video keyframes, labeled Image 1 through Image {n} "
        f"in the order given. For each image, write one concise English sentence "
        f"describing the main subject, action, and setting. Do not mention "
        f"timestamps, frame numbers, or that these are video frames.\n"
        f"Respond with exactly {n} lines, each in the form 'Image i: <caption>'."
    )


def resize_and_encode(image_path: Path, max_side: int, jpeg_quality: int) -> tuple[str, imagehash.ImageHash]:
    """Resize to max_side (keeping aspect ratio), strip EXIF, re-encode as
    JPEG, and return (base64 string, perceptual hash of the resized image)."""
    im = Image.open(image_path).convert("RGB")
    scale = max_side / max(im.size)
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)

    phash = imagehash.phash(im, hash_size=8)

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii"), phash


def _call_ollama(session: requests.Session, images_b64: list[str], prompt: str, model: str,
                  ollama_url: str, num_predict: int, keep_alive: str) -> dict:
    response = session.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"num_predict": num_predict, "temperature": 0.3},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def generate_caption_single(
    session: requests.Session,
    image_b64: str,
    model: str,
    ollama_url: str,
    num_predict: int,
    keep_alive: str,
    max_retries: int = 2,
) -> str:
    budget = num_predict
    data = _call_ollama(session, [image_b64], SINGLE_PROMPT, model, ollama_url, budget, keep_alive)
    retries = 0
    while data.get("done_reason") == "length" and retries < max_retries:
        budget *= 2
        retries += 1
        data = _call_ollama(session, [image_b64], SINGLE_PROMPT, model, ollama_url, budget, keep_alive)
    return data["response"].strip()


def generate_captions_batch(
    session: requests.Session,
    images_b64: list[str],
    model: str,
    ollama_url: str,
    num_predict_base: int,
    keep_alive: str,
    max_retries: int = 2,
) -> list[str] | None:
    """Caption a batch of images in one call. Returns a list of captions
    (same length/order as images_b64) on success, or None if the batch
    couldn't be parsed/completed after retries -- caller should fall back to
    per-image captioning in that case."""
    n = len(images_b64)
    prompt = build_batch_prompt(n)
    budget = num_predict_base * n

    for _ in range(max_retries + 1):
        data = _call_ollama(session, images_b64, prompt, model, ollama_url, budget, keep_alive)
        truncated = data.get("done_reason") == "length"
        captions = _parse_batch_response(data["response"], n)
        if captions is not None and not truncated:
            return captions
        budget *= 2

    return None


def _parse_batch_response(response: str, n: int) -> list[str] | None:
    by_index: dict[int, str] = {}
    for line in response.splitlines():
        m = BATCH_LINE_RE.match(line)
        if m:
            idx = int(m.group(1))
            by_index[idx] = m.group(2).strip()
    if len(by_index) != n or set(by_index) != set(range(1, n + 1)):
        return None
    return [by_index[i] for i in range(1, n + 1)]


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
    max_side: int,
    jpeg_quality: int,
    batch_size: int,
    phash_threshold: int,
    dedup: bool,
) -> None:
    json_path = json_dir / f"{video_id}.json"
    keyframe_dir = keyframe_root / video_id

    data = json.loads(json_path.read_text())
    keyframes = data["keyframes"]
    if limit is not None:
        keyframes = keyframes[:limit]

    pending = [kf for kf in keyframes if overwrite or not kf.get("caption")]
    skipped = len(keyframes) - len(pending)
    if skipped:
        print(f"[caption-fast] resuming: {skipped} keyframes already captioned, skipping")

    total = len(pending)
    print(f"[caption-fast] {video_id}: {total} keyframes to caption, model={model}, "
          f"max_side={max_side}, batch_size={batch_size}, dedup={dedup}")

    session = requests.Session()
    seen_hashes: list[tuple[imagehash.ImageHash, str]] = []  # (phash, caption) of already-captioned frames
    n_written = 0
    n_dedup_hits = 0
    n_batches_fallback = 0
    t_start = time.time()

    def checkpoint_if_due():
        nonlocal n_written
        if n_written % checkpoint_every == 0:
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"  [checkpoint] wrote progress ({n_written}/{total}) to {json_path}")

    i = 0
    while i < total:
        # Resize + hash the next chunk, splitting off any near-duplicates of
        # an already-captioned frame so they don't consume a batch slot.
        chunk_kfs = []
        chunk_b64 = []
        while i < total and len(chunk_kfs) < batch_size:
            kf = pending[i]
            image_path = keyframe_dir / f"{kf['n']:03d}.jpg"
            img_b64, phash = resize_and_encode(image_path, max_side, jpeg_quality)

            reused = None
            if dedup:
                for seen_hash, seen_caption in seen_hashes:
                    if phash - seen_hash <= phash_threshold:
                        reused = seen_caption
                        break

            if reused is not None:
                kf["caption"] = reused
                n_dedup_hits += 1
                n_written += 1
                print(f"[{n_written}/{total}] n={kf['n']:03d} frame_idx={kf['frame_idx']} "
                      f"(dedup) -> {kf['caption']}")
                checkpoint_if_due()
            else:
                chunk_kfs.append(kf)
                chunk_b64.append((img_b64, phash))
            i += 1

        if not chunk_kfs:
            continue

        chunk_n = len(chunk_kfs)
        t0 = time.time()
        captions = None
        used_batch = False
        if chunk_n > 1:
            captions = generate_captions_batch(
                session, [b for b, _ in chunk_b64], model, ollama_url, num_predict, keep_alive)
            if captions is not None:
                used_batch = True
            else:
                n_batches_fallback += 1
                print(f"  [warn] batch of {chunk_n} frames failed to parse/complete, "
                      f"falling back to per-image calls")

        if captions is None:
            captions = [
                generate_caption_single(session, b, model, ollama_url, num_predict, keep_alive)
                for b, _ in chunk_b64
            ]

        dt = time.time() - t0
        if chunk_n > 1:
            mode = "batch call" if used_batch else "per-image fallback"
            print(f"  [{mode}] {chunk_n} frames in this group, {dt:.1f}s total "
                  f"({dt / chunk_n:.1f}s/frame avg)")

        for kf, caption, (_, phash) in zip(chunk_kfs, captions, chunk_b64):
            kf["caption"] = caption
            seen_hashes.append((phash, caption))
            n_written += 1
            timing = f"{dt:.1f}s total for {chunk_n} frames" if chunk_n > 1 else f"{dt:.1f}s"
            print(f"[{n_written}/{total}] n={kf['n']:03d} frame_idx={kf['frame_idx']} "
                  f"({timing}) -> {kf['caption']}")
            checkpoint_if_due()

    elapsed = time.time() - t_start
    if total:
        print(f"Done: {total} captions in {elapsed:.1f}s ({elapsed / total:.2f}s/keyframe avg), "
              f"{n_dedup_hits} reused via dedup, {n_batches_fallback} batches fell back to per-image")
    else:
        print("Done: nothing to caption (all keyframes already had captions)")

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Wrote captions into {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR)
    parser.add_argument("--keyframe-root", default=DEFAULT_KEYFRAME_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--num-predict", type=int, default=400,
                         help="Base num_predict per image (scaled up for batches)")
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-side", type=int, default=448,
                         help="Resize longest image side to this many pixels (default: 448)")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--batch-size", type=int, default=1,
                         help="Images per Ollama call (default: 1 = disabled. qwen3-vl was "
                              "observed to be unreliable/slow at following the "
                              "'Image i: <caption>' format for batches >1 -- large num_predict "
                              "budgets for a failing batch can hang for a long time before "
                              "falling back to per-image calls. Try >1 only if you want to "
                              "experiment and are prepared for that cost.")
    parser.add_argument("--phash-threshold", type=int, default=8,
                         help="Max Hamming distance to consider two frames near-duplicates "
                              "(same convention as scripts/transnetv2_dake_keyframes.py)")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                         help="Disable perceptual-hash dedup (default: enabled)")
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
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        batch_size=args.batch_size,
        phash_threshold=args.phash_threshold,
        dedup=args.dedup,
    )
