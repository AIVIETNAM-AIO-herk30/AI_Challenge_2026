"""
Convert organizer-provided map-keyframes CSVs to one JSON file per video.

Input:  data/raw/queries/map-keyframes-aic25-b1/map-keyframes/{video_id}.csv
        columns: n, pts_time, fps, frame_idx
Output: data/json/{video_id}.json
        {"video_id": ..., "num_keyframes": ..., "keyframes": [
            {"n": 1, "frame_idx": 0, "pts_time": 0.0, "fps": 30.0,
             "frame_id": "{video_id}_{frame_idx:06d}"},
            ...
        ]}

The frame_id field follows the f"{video_id}_{frame_idx:06d}" convention used
elsewhere in this codebase (es_store.py, transnetv2_dake_keyframes.py), so
downstream code can join this file against other per-keyframe JSON metadata
by frame_id alone.
"""
import argparse
import csv
import json
from pathlib import Path

DEFAULT_INPUT_DIR = "data/raw/queries/map-keyframes-aic25-b1/map-keyframes"
DEFAULT_OUTPUT_DIR = "data/json"


def convert_one(csv_path: Path, output_dir: Path) -> int:
    video_id = csv_path.stem
    keyframes = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame_idx = int(row["frame_idx"])
            keyframes.append({
                "n": int(row["n"]),
                "frame_idx": frame_idx,
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "frame_id": f"{video_id}_{frame_idx:06d}",
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.json"
    with output_path.open("w") as f:
        json.dump({
            "video_id": video_id,
            "num_keyframes": len(keyframes),
            "keyframes": keyframes,
        }, f, indent=2)

    return len(keyframes)


def run(input_dir: Path, output_dir: Path) -> None:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {input_dir}")
        return

    total = len(csv_paths)
    for i, csv_path in enumerate(csv_paths, start=1):
        n_keyframes = convert_one(csv_path, output_dir)
        print(f"[{i}/{total}] {csv_path.stem}: {n_keyframes} keyframes -> "
              f"{output_dir / (csv_path.stem + '.json')}")

    print(f"Done: converted {total} videos into {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                         help=f"Directory of map-keyframes CSVs (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help=f"Directory to write per-video JSON files (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    run(Path(args.input_dir), Path(args.output_dir))
