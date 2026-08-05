#!/usr/bin/env python3
"""Run the integrated pipeline with the standard local args (no manual cd/activate).

Usage:
    ./main.py                     # uses the default video dir below
    ./main.py <video_dir_or_mp4>  # overrides --video-dir for this run
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data/raw/queries/Videos_L21_a/video"

if len(sys.argv) > 1:
    arg_path = Path(sys.argv[1]).expanduser()
    if not arg_path.is_absolute():
        arg_path = Path.cwd() / arg_path
    # integrated_pipeline.py globs *.mp4 inside --video-dir; if a single video
    # file is passed, point --video-dir at its parent folder instead.
    VIDEO_DIR = arg_path.parent if arg_path.suffix.lower() == ".mp4" else arg_path
else:
    VIDEO_DIR = DEFAULT_VIDEO_DIR

cmd = [
    str(VENV_PYTHON), "-m", "scripts.integrated_pipeline",
    "--video-dir", str(VIDEO_DIR),
    "--asr-dir", "data/asr",
    "--keyframe-dir", "data/keyframe",
    "--output-dir", "data/fullRecap",
    "--model-dir", "weights/transnetv2/",
]

env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"

sys.exit(subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode)
