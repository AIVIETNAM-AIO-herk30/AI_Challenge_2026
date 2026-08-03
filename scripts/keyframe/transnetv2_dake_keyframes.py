"""
Standalone TransNetV2 + DAKE keyframe extraction demo.

Single-file, no imports from src/ -- deliberately decoupled from the main
package so it can be run/tuned independently while the team decides whether
to fold this into src/retrieval/.

Pipeline:
  1. TransNetV2 detects shot boundaries (same model/approach as
     src/retrieval/shot_detector.py): "where does the camera cut?"
  2. DAKE (Dynamic-Aware Keyframe Extraction) selects one-or-more keyframes
     *within* each shot: "which moments in this continuous shot actually
     matter?" -- Vo et al., "U-CESE: Unified Clip-based Event Search Engine
     for AI Challenge HCMC 2025" (arXiv:2605.23274), Sec. 4.1.1.

===============================================================================
CHANGELOG (v2) -- xem chat để biết bối cảnh đầy đủ, tóm tắt ở đây:
===============================================================================

VẤN ĐỀ QUAN SÁT ĐƯỢC: trên cảnh talking-head gần như tĩnh (2 MC đứng yên,
background 3D render dạng gradient hoàng hôn), DAKE bản v1 vẫn chọn ra nhiều
keyframe gần như giống hệt nhau trong cùng vài giây. Nguyên nhân: JPEG
byte-size (tín hiệu steepness duy nhất ở bản v1) nhạy với NOISE NÉN ở vùng
gradient/render 3D và với overlay UI (đồng hồ góc phải tick mỗi giây, banner
tên MC animate-in) -- những thứ này làm JPEG size dao động dù mắt người thấy
cảnh không đổi gì đáng kể.

BA THAY ĐỔI CHÍNH:

  1. mask_broadcast_overlay(): che vùng overlay cố định (logo+giờ góc phải,
     dải banner tên MC giữa khung) TRƯỚC KHI tính cả 3 tín hiệu (steepness,
     pHash, color-hist) -- nếu không che, đồng hồ/banner bị hiểu nhầm là
     "nội dung cảnh quay thay đổi". Tọa độ mask hiện hardcode theo layout
     HTV9 trong ảnh bạn gửi -- cần chỉnh lại nếu đổi sang chương trình có
     layout khác, hoặc tắt hẳn bằng --disable-overlay-mask.

  2. Thêm pHash (frame_phash) làm gate lọc trùng lặp về CẤU TRÚC/ĐỘ SÁNG:
     pHash resize ảnh nhỏ + chuyển grayscale + DCT, giữ hệ số tần số thấp
     -- đo "bố cục ảnh có giống nhau không", KHÔNG đo màu sắc (grayscale đã
     vứt bỏ hue/saturation).

  3. Thêm color histogram (HSV, color_hist_distance) làm gate lọc trùng lặp
     riêng về MÀU SẮC -- bổ sung đúng phần pHash bỏ sót. 2 gate này dùng
     logic OR khi giữ frame (AND khi loại): một candidate chỉ bị coi là
     trùng lặp nếu NÓ VỪA giống cấu trúc VỪA giống màu so với 1 keyframe đã
     chọn trong cùng shot -- khác nhau ở MỘT trong hai (ví dụ đổi màu áo
     nhưng bố cục y hệt, hoặc đổi bố cục nhưng tông màu y hệt) thì vẫn giữ,
     vì cả hai đều có thể là tín hiệu retrieval quan trọng (query theo màu
     sắc, ví dụ "người mặc áo đỏ", vẫn cần phân biệt được).

Cả 3 tín hiệu được tính trong CÙNG một lượt decode ở Pass 1 (không thêm lượt
đọc video nào) -- tận dụng pixel buffer đã có sẵn trước khi nó bị JPEG-encode
và discard như bản v1.

Two sequential decode passes over the video (never index-seeks, matching
video_indexer.py's own rationale: cv2.CAP_PROP_POS_FRAMES isn't reliably
frame-accurate across all H.264 GOP structures):
  Pass 1: decode every frame once; for each frame, compute (a) JPEG-encoded
          size, (b) perceptual hash, (c) HSV color histogram -- all from the
          same in-memory pixel buffer -- then discard the pixel buffer.
  Pass 2: decode again, this time saving only the frames DAKE selected.

Output (per keyframe, using the video-wide real frame_idx -- not a
keyframe sequence number):
  {output_dir}/{video_id}/{frame_id}.jpg
  {output_dir}/{video_id}/{frame_id}.json   <- one metadata file per keyframe
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# THÊM MỚI: cần cho pHash. Cài bằng: pip install imagehash --break-system-packages
import imagehash
from PIL import Image


@dataclass
class Shot:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or np.isnan(fps) or fps <= 0:
        raise ValueError(f"Could not read a valid FPS for {video_path}")
    return fps


def detect_shots(
    video_path: Path,
    model_dir: Path,
    threshold: float = 0.5,
    min_shot_duration_sec: float = 0.5,
) -> list[Shot]:
    """TransNetV2 shot-boundary detection -- same model/config keys as
    src/retrieval/shot_detector.py, reimplemented inline here since this
    file is meant to stand alone."""
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # tiny model, avoid TF grabbing VRAM PyTorch needs
    from transnetv2 import TransNetV2

    model = TransNetV2(model_dir=str(model_dir))
    native_fps = get_fps(video_path)
    _frames, single_frame_pred, _all_frame_pred = model.predict_video(str(video_path))
    scenes = model.predictions_to_scenes(single_frame_pred, threshold=threshold)

    shots = []
    for start_frame, end_frame in scenes:
        start_frame, end_frame = int(start_frame), int(end_frame)
        start_sec = start_frame / native_fps
        end_sec = end_frame / native_fps
        if end_sec - start_sec < min_shot_duration_sec:
            continue
        shots.append(Shot(start_frame, end_frame, start_sec, end_sec))
    return shots


# =============================================================================
# THÊM MỚI: che vùng overlay cố định (logo/giờ + banner tên MC) trước khi
# tính bất kỳ tín hiệu nào -- xem CHANGELOG #1 ở đầu file.
# Tọa độ hardcode theo layout chương trình "60 giây" / HTV9 trong ảnh bạn gửi.
# Nếu dùng dataset có layout khác, chỉnh lại 2 rect này hoặc truyền
# --disable-overlay-mask để tắt hẳn bước này.
# =============================================================================
def mask_broadcast_overlay(frame_bgr: np.ndarray) -> np.ndarray:
    """Trả về BẢN SAO đã che (không sửa frame gốc) -- gán 0 (đen) vào vùng
    logo kênh + đồng hồ (góc phải trên) và vùng banner tên MC (dải ngang
    giữa khung, phía trên bàn tin)."""
    h, w = frame_bgr.shape[:2]
    masked = frame_bgr.copy()
    masked[0 : int(h * 0.15), int(w * 0.85) : w] = 0  # logo kênh + giờ, góc phải trên
    masked[int(h * 0.53) : int(h * 0.63), 0:w] = 0  # dải banner tên MC, giữa khung
    return masked


def jpeg_size(frame_bgr: np.ndarray, quality: int = 90) -> int:
    """JPEG-encode a frame in-memory and return its compressed size in
    bytes -- DAKE's motion proxy: dynamic content compresses less
    predictably (larger, more variable size) than static content."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return int(buf.size)


# THÊM MỚI: pHash -- đo độ khác biệt CẤU TRÚC/ĐỘ SÁNG (grayscale, không màu).
def frame_phash(frame_bgr: np.ndarray, hash_size: int = 8) -> imagehash.ImageHash:
    """pHash (DCT-based perceptual hash). Chuyển grayscale trước khi hash --
    nghĩa là nó KHÔNG đo màu sắc, chỉ đo bố cục/độ sáng. Bổ sung bằng
    color_histogram() bên dưới cho phần màu sắc còn thiếu."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(frame_rgb), hash_size=hash_size)


# THÊM MỚI: color histogram (HSV) -- đo độ khác biệt MÀU SẮC, phần pHash bỏ sót.
def color_histogram(frame_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """Histogram màu trong không gian HSV (tách hue khỏi độ sáng), đã
    normalize -- dùng với color_hist_distance() bên dưới."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def color_hist_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Bhattacharyya distance: 0 = màu giống hệt, càng gần 1 = càng khác
    màu. Cần tự tune threshold trên vài cặp ảnh thật (xem gợi ý ở argparse
    --dake-color-dup-threshold)."""
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


# =============================================================================
# THAY ĐỔI: measure_frame_sizes() (v1) -> measure_frame_features() (v2).
# Vẫn 1 lượt decode duy nhất, nhưng giờ tính CẢ BA tín hiệu (size, phash,
# color-hist) từ CÙNG một pixel buffer trước khi discard, và áp dụng mask
# overlay trước khi tính bất kỳ tín hiệu nào. Thêm progress log mỗi
# `log_every` frame để không bị tưởng nhầm là treo trên video dài.
# =============================================================================
def measure_frame_features(
    video_path: Path,
    max_frame_idx: int,
    jpeg_quality: int,
    phash_size: int,
    color_bins: int,
    apply_overlay_mask: bool,
    log_every: int = 5000,
) -> tuple[dict[int, int], dict[int, imagehash.ImageHash], dict[int, np.ndarray]]:
    """Pass 1: single sequential decode. For each frame, compute JPEG size,
    perceptual hash, and color histogram, then discard the pixel buffer."""
    cap = cv2.VideoCapture(str(video_path))
    sizes: dict[int, int] = {}
    phashes: dict[int, imagehash.ImageHash] = {}
    color_hists: dict[int, np.ndarray] = {}
    idx = 0
    try:
        while idx <= max_frame_idx:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_for_signals = mask_broadcast_overlay(frame_bgr) if apply_overlay_mask else frame_bgr

            sizes[idx] = jpeg_size(frame_for_signals, jpeg_quality)
            phashes[idx] = frame_phash(frame_for_signals, phash_size)
            color_hists[idx] = color_histogram(frame_for_signals, color_bins)

            idx += 1
            if idx % log_every == 0:
                print(f"      ... {idx}/{max_frame_idx + 1} frames measured")
    finally:
        cap.release()
    return sizes, phashes, color_hists


# =============================================================================
# THAY ĐỔI: dake_select_keyframes() -- thêm 2 tham số mới (frame_phashes,
# frame_color_hists) và gate loại trùng lặp mới. Logic min_gap_frames (đã
# thêm ở bản trước) giữ nguyên không đổi.
# =============================================================================
def dake_select_keyframes(
    frame_sizes: dict[int, int],
    frame_phashes: dict[int, imagehash.ImageHash],
    frame_color_hists: dict[int, np.ndarray],
    shot: Shot,
    window: int,
    keyframe_ratio: float,
    min_keyframes_per_shot: int,
    min_gap_frames: int,
    phash_dup_hamming_threshold: int,
    color_dup_bhattacharyya_threshold: float,
) -> list[dict]:
    """DAKE: rank candidate frames trong 1 shot theo aggregated JPEG-size
    steepness (như v1, không đổi), nhưng giờ chỉ NHẬN 1 candidate nếu nó đủ
    khác biệt so với MỌI keyframe đã chọn trong cùng shot -- theo CẢ HAI
    tiêu chí min_gap_frames (khoảng cách thời gian) VÀ is_duplicate (cấu
    trúc + màu, xem bên dưới).

    is_duplicate = (phash gần)  AND  (color-hist gần)
    -- tức là bị loại chỉ khi giống CẢ cấu trúc LẪN màu; khác 1 trong 2 thì
    vẫn được giữ (xem CHANGELOG #3 ở đầu file)."""
    indices = [i for i in range(shot.start_frame, shot.end_frame + 1) if i in frame_sizes]
    if len(indices) <= min_keyframes_per_shot:
        return [
            {"frame_idx": i, "aggregated_steepness_score": 0.0, "rank_within_shot": r + 1}
            for r, i in enumerate(indices)
        ]

    steepness: dict[tuple[int, int], float] = {}
    for a, b in zip(indices, indices[1:]):
        sa, sb = frame_sizes[a], frame_sizes[b]
        steepness[(a, b)] = abs(sa - sb) / max(sa, sb, 1)

    aggregated: dict[int, float] = {}
    for pos, i in enumerate(indices):
        neighbours = indices[max(0, pos - window) : pos + window + 1]
        aggregated[i] = sum(
            steepness.get((a, b), 0.0) for a, b in zip(neighbours, neighbours[1:])
        )

    n_keyframes = max(min_keyframes_per_shot, math.ceil(keyframe_ratio * len(indices)))
    candidates = sorted(indices, key=lambda i: aggregated[i], reverse=True)

    selected: list[int] = []
    rank_of: dict[int, int] = {}
    for i in candidates:
        if len(selected) >= n_keyframes:
            break

        if any(abs(i - s) < min_gap_frames for s in selected):
            continue

        # THÊM MỚI: gate lọc trùng lặp theo cấu trúc + màu.
        is_duplicate_of_existing = any(
            (frame_phashes[i] - frame_phashes[s] <= phash_dup_hamming_threshold)
            and (color_hist_distance(frame_color_hists[i], frame_color_hists[s]) <= color_dup_bhattacharyya_threshold)
            for s in selected
        )
        if is_duplicate_of_existing:
            continue

        rank_of[i] = len(selected) + 1
        selected.append(i)

    selected_sorted = sorted(selected)
    return [
        {
            "frame_idx": i,
            "aggregated_steepness_score": round(aggregated[i], 6),
            "rank_within_shot": rank_of[i],
        }
        for i in selected_sorted
    ]


def extract_and_save_keyframes(
    video_path: Path,
    selected: dict[int, dict],  # frame_idx -> metadata (without image_path/frame_id yet)
    video_id: str,
    native_fps: float,
    output_dir: Path,
) -> list[str]:
    """Pass 2: single sequential decode, save only the DAKE-selected frames
    (as JPEGs) plus one JSON metadata file per keyframe."""
    out_dir = output_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_ids = []
    idx = 0
    remaining = set(selected)
    try:
        while remaining:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx in remaining:
                frame_id = f"{video_id}_{idx:06d}"
                jpg_path = out_dir / f"{frame_id}.jpg"
                json_path = out_dir / f"{frame_id}.json"

                # Lưu ý: ảnh lưu ra KHÔNG bị che overlay -- mask chỉ dùng nội
                # bộ để tính tín hiệu so sánh, không áp dụng lên ảnh output.
                cv2.imwrite(str(jpg_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

                record = dict(selected[idx])
                record.update(
                    {
                        "frame_id": frame_id,
                        "video_id": video_id,
                        "frame_idx": idx,
                        "timestamp_sec": round(idx / native_fps, 3),
                        "fps": native_fps,
                        "keyframe_method": "TransNetV2+DAKE+pHash+ColorHist",
                        "image_path": str(jpg_path),
                    }
                )
                json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
                frame_ids.append(frame_id)
                remaining.discard(idx)
            idx += 1
    finally:
        cap.release()
    return frame_ids


def run(
    video_path: Path,
    output_dir: Path,
    model_dir: Path,
    threshold: float,
    min_shot_duration_sec: float,
    dake_window: int,
    dake_keyframe_ratio: float,
    dake_min_keyframes_per_shot: int,
    dake_min_gap_sec: float,
    jpeg_quality: int,
    phash_size: int,
    phash_dup_hamming_threshold: int,
    color_bins: int,
    color_dup_bhattacharyya_threshold: float,
    apply_overlay_mask: bool,
) -> None:
    video_id = video_path.stem
    native_fps = get_fps(video_path)
    min_gap_frames = max(1, round(dake_min_gap_sec * native_fps))

    print(f"[1/4] TransNetV2 shot detection on {video_path.name} ...")
    shots = detect_shots(video_path, model_dir, threshold, min_shot_duration_sec)
    print(f"      -> {len(shots)} shots")

    max_frame_idx = max((s.end_frame for s in shots), default=0)
    print(
        f"[2/4] Pass 1: measuring JPEG-size / pHash / color-hist for frames "
        f"0..{max_frame_idx} (overlay mask={'on' if apply_overlay_mask else 'off'}) ..."
    )
    frame_sizes, frame_phashes, frame_color_hists = measure_frame_features(
        video_path, max_frame_idx, jpeg_quality, phash_size, color_bins, apply_overlay_mask
    )
    print(f"      -> measured {len(frame_sizes)} frames")

    print(
        f"[3/4] DAKE: selecting keyframes per shot "
        f"(min_gap={min_gap_frames} frames, phash_thresh={phash_dup_hamming_threshold}, "
        f"color_thresh={color_dup_bhattacharyya_threshold}) ..."
    )
    selected: dict[int, dict] = {}
    for shot_index, shot in enumerate(shots):
        picks = dake_select_keyframes(
            frame_sizes,
            frame_phashes,
            frame_color_hists,
            shot,
            dake_window,
            dake_keyframe_ratio,
            dake_min_keyframes_per_shot,
            min_gap_frames,
            phash_dup_hamming_threshold,
            color_dup_bhattacharyya_threshold,
        )
        for pick in picks:
            selected[pick["frame_idx"]] = {
                **pick,
                "shot_index": shot_index,
                "shot_start_frame": shot.start_frame,
                "shot_end_frame": shot.end_frame,
                "shot_start_sec": round(shot.start_sec, 3),
                "shot_end_sec": round(shot.end_sec, 3),
                "dake_window": dake_window,
                "dake_keyframe_ratio": dake_keyframe_ratio,
                "dake_min_gap_sec": dake_min_gap_sec,
                "phash_dup_hamming_threshold": phash_dup_hamming_threshold,
                "color_dup_bhattacharyya_threshold": color_dup_bhattacharyya_threshold,
            }
    print(f"      -> {len(selected)} keyframes selected across {len(shots)} shots")

    print(f"[4/4] Pass 2: saving keyframe images + per-keyframe JSON to {output_dir / video_id} ...")
    frame_ids = extract_and_save_keyframes(video_path, selected, video_id, native_fps, output_dir)
    print(f"      -> wrote {len(frame_ids)} keyframe .jpg + .json pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to the source .mp4")
    parser.add_argument("--output-dir", type=Path, default=Path("data/keyframe"))
    parser.add_argument("--model-dir", type=Path, default=Path("weights/transnetv2/"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-shot-duration-sec", type=float, default=2)
    parser.add_argument("--dake-window", type=int, default=5, help="Steepness aggregation window (frames)")
    parser.add_argument("--dake-keyframe-ratio", type=float, default=0.05, help="Fraction of a shot's frames to keep")
    parser.add_argument("--dake-min-keyframes-per-shot", type=int, default=1)
    parser.add_argument(
        "--dake-min-gap-sec",
        type=float,
        default=0.5,
        help="Minimum time gap enforced between two selected keyframes within the same shot",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)

    # THÊM MỚI: các tham số cho pHash + color-hist gate.
    parser.add_argument(
        "--dake-phash-size", type=int, default=8, dest="phash_size",
        help="Kích thước hash pHash (8 -> hash 64-bit). Tăng lên (vd 16) để nhạy hơn với chi tiết nhỏ.",
    )
    parser.add_argument(
        "--dake-phash-dup-threshold", type=int, default=8, dest="phash_dup_hamming_threshold",
        help="Ngưỡng Hamming distance coi là 'giống cấu trúc'. Thấp = khắt khe (giữ nhiều keyframe hơn). "
        "Nên tự đo trên vài cặp ảnh thật trước khi chốt số (xem gợi ý trong chat).",
    )
    parser.add_argument(
        "--dake-color-bins", type=int, default=8, dest="color_bins",
        help="Số bin mỗi kênh H/S/V cho color histogram (tổng bins = n^3).",
    )
    parser.add_argument(
        "--dake-color-dup-threshold", type=float, default=0.3, dest="color_dup_bhattacharyya_threshold",
        help="Ngưỡng Bhattacharyya distance coi là 'giống màu' (0=giống hệt, 1=khác hoàn toàn). Cần tự tune.",
    )
    parser.add_argument(
        "--disable-overlay-mask", action="store_true",
        help="Tắt việc che vùng logo/giờ + banner tên MC trước khi tính tín hiệu -- bật cờ này nếu "
        "dùng dataset không có layout overlay giống HTV9, hoặc muốn so sánh có/không mask.",
    )
    args = parser.parse_args()

    run(
        video_path=args.video,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        threshold=args.threshold,
        min_shot_duration_sec=args.min_shot_duration_sec,
        dake_window=args.dake_window,
        dake_keyframe_ratio=args.dake_keyframe_ratio,
        dake_min_keyframes_per_shot=args.dake_min_keyframes_per_shot,
        dake_min_gap_sec=args.dake_min_gap_sec,
        jpeg_quality=args.jpeg_quality,
        phash_size=args.phash_size,
        phash_dup_hamming_threshold=args.phash_dup_hamming_threshold,
        color_bins=args.color_bins,
        color_dup_bhattacharyya_threshold=args.color_dup_bhattacharyya_threshold,
        apply_overlay_mask=not args.disable_overlay_mask,
    )