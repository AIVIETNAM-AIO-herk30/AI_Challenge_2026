"""
Fetch YouTube transcripts (Bước A) for AIC 2026 batch-1 videos.

Reads watch_url from the organizer-provided per-video metadata
(media-info-aic25-b1), tries the YouTube Transcript API for a Vietnamese
transcript (manually created first, then auto-generated), and writes a
unified schema to data/asr/{video_id}.json on success.

Videos that fail for a permanent, per-video reason (subtitles disabled,
unplayable, age-restricted, no vi transcript at all) are simply left without
an output file -- scripts/batch_transcribe_whisperx.py (Bước B) finds them by
scanning for videos missing a data/asr/{video_id}.json.

RequestBlocked/IpBlocked is NOT a per-video problem -- it means this IP is
currently rate-limited by YouTube for the whole batch. On that error we do
NOT advance to the next video or mark anything for fallback; we sleep (with
exponential backoff) and retry the SAME video to confirm the block lifted
before resuming.

Requires: pip install youtube-transcript-api
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,  # IpBlocked subclasses this -- same except clause catches both
    TranscriptsDisabled,
    VideoUnplayable,
    YouTubeTranscriptApi,
)

DEFAULT_MEDIA_INFO_DIR = "data/raw/queries/media-info-aic25-b1/media-info"
DEFAULT_ASR_DIR = "data/asr"
WATCH_URL_RE = re.compile(r"[?&]v=([\w-]+)")

# Priority per spec 2.1: Vietnamese only (manual vi, then auto-generated vi).
# No English fallback -- back-translating an already-noisy auto-caption loses
# too much accuracy for Vietnamese TV-news content.
LANGUAGES = ["vi"]


def parse_youtube_id(watch_url: str) -> str | None:
    m = WATCH_URL_RE.search(watch_url)
    return m.group(1) if m else None


def fetch_one(api: YouTubeTranscriptApi, youtube_id: str) -> tuple[list[dict], str, bool]:
    """Returns (raw_segments, language_code, is_generated). Raises the
    library's exceptions on failure -- caller classifies them."""
    transcript_list = api.list(youtube_id)
    try:
        transcript = transcript_list.find_manually_created_transcript(LANGUAGES)
    except NoTranscriptFound:
        transcript = transcript_list.find_generated_transcript(LANGUAGES)

    fetched = transcript.fetch()
    raw_segments = [
        {"start": s.start, "end": s.start + s.duration, "text": s.text}
        for s in fetched
    ]
    return raw_segments, transcript.language_code, transcript.is_generated


def run(
    media_info_dir: Path,
    asr_dir: Path,
    video_ids: list[str] | None,
    limit: int | None,
    sleep_min: float,
    sleep_max: float,
    block_sleep_minutes: float,
    block_sleep_max_minutes: float,
) -> None:
    asr_dir.mkdir(parents=True, exist_ok=True)

    if video_ids:
        media_paths = [media_info_dir / f"{vid}.json" for vid in video_ids]
    else:
        media_paths = sorted(media_info_dir.glob("*.json"))
    if limit is not None:
        media_paths = media_paths[:limit]

    api = YouTubeTranscriptApi()
    total = len(media_paths)
    n_ok = n_fallback = n_skipped = 0
    current_block_sleep = block_sleep_minutes

    i = 0
    while i < len(media_paths):
        media_path = media_paths[i]
        video_id = media_path.stem
        out_path = asr_dir / f"{video_id}.json"

        if out_path.exists():
            n_skipped += 1
            i += 1
            continue

        media = json.loads(media_path.read_text())
        watch_url = media.get("watch_url", "")
        youtube_id = parse_youtube_id(watch_url)
        if youtube_id is None:
            print(f"[{i + 1}/{total}] {video_id}: could not parse youtube_video_id from "
                  f"watch_url={watch_url!r} -- needs Bước B")
            n_fallback += 1
            i += 1
            continue

        try:
            raw_segments, language_code, is_generated = fetch_one(api, youtube_id)
            out_path.write_text(json.dumps({
                "video_id": video_id,
                "transcript_source": "youtube_captions",
                "youtube_video_id": youtube_id,
                "language": language_code,
                "is_auto_generated": is_generated,
                "raw_segments": raw_segments,
                "intervals": [],
            }, indent=2, ensure_ascii=False))
            print(f"[{i + 1}/{total}] {video_id} (yt={youtube_id}): "
                  f"{len(raw_segments)} segments, lang={language_code}, auto={is_generated}")
            n_ok += 1
            current_block_sleep = block_sleep_minutes  # reset backoff after success
            i += 1

        except RequestBlocked as e:
            print(f"[{i + 1}/{total}] {video_id}: {type(e).__name__} -- YouTube is "
                  f"rate-limiting/blocking this IP for the whole batch. Sleeping "
                  f"{current_block_sleep:.0f}min before retrying THIS SAME video "
                  f"(not advancing, not marking for fallback)...")
            time.sleep(current_block_sleep * 60)
            current_block_sleep = min(current_block_sleep * 2, block_sleep_max_minutes)
            continue  # retry the same video_id/index

        except (TranscriptsDisabled, VideoUnplayable, AgeRestricted, NoTranscriptFound) as e:
            print(f"[{i + 1}/{total}] {video_id}: {type(e).__name__} -- permanent for this "
                  f"video, needs Bước B (WhisperX)")
            n_fallback += 1
            i += 1

        except CouldNotRetrieveTranscript as e:
            # Anything else the library can raise (VideoUnavailable, PoTokenRequired,
            # YouTubeRequestFailed, ...) that isn't in the spec's table -- treat as
            # needs-fallback rather than silently skipping or crashing the batch.
            print(f"[{i + 1}/{total}] {video_id}: unclassified {type(e).__name__}: {e} "
                  f"-- treating as needs Bước B")
            n_fallback += 1
            i += 1

        time.sleep(random.uniform(sleep_min, sleep_max))

    print(f"Done: {n_ok} fetched via youtube_captions, {n_fallback} need Bước B "
          f"(WhisperX), {n_skipped} already had output, out of {total} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-info-dir", default=DEFAULT_MEDIA_INFO_DIR)
    parser.add_argument("--asr-dir", default=DEFAULT_ASR_DIR)
    parser.add_argument("--video-ids", nargs="+", default=None,
                         help="Only process these video_ids (space-separated)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-min", type=float, default=1.5)
    parser.add_argument("--sleep-max", type=float, default=3.0)
    parser.add_argument("--block-sleep-minutes", type=float, default=20.0,
                         help="Initial sleep before retrying after RequestBlocked/"
                              "IpBlocked (default: 20 min)")
    parser.add_argument("--block-sleep-max-minutes", type=float, default=120.0,
                         help="Cap for the exponential backoff on repeated blocks "
                              "(default: 120 min)")
    args = parser.parse_args()

    run(
        media_info_dir=Path(args.media_info_dir),
        asr_dir=Path(args.asr_dir),
        video_ids=args.video_ids,
        limit=args.limit,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        block_sleep_minutes=args.block_sleep_minutes,
        block_sleep_max_minutes=args.block_sleep_max_minutes,
    )
