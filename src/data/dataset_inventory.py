"""
Dataset inventory and auditing module for VDzSL.

Provides fast, multi-threaded sequential video validation, metadata extraction,
and manifest generation with strict UTF-8 handling.
"""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import cv2
import pandas as pd
from tqdm import tqdm

from src.data.label_parser import LabelParser


# Camera angle polar coordinate mapping based on VDzSL specification
CAMERA_POLAR_COORDS = {
    "cam0": {"theta": 0, "phi": 0, "description": "Frontal view (0 deg horiz, 0 deg vert)"},
    "cam1": {"theta": -30, "phi": 0, "description": "Left horizontal angle (-30 deg)"},
    "cam2": {"theta": 30, "phi": 0, "description": "Right horizontal angle (+30 deg)"},
    "cam3": {"theta": 0, "phi": 15, "description": "Elevated vertical angle (+15 deg)"},
    "cam4": {"theta": 0, "phi": -15, "description": "Depressed vertical angle (-15 deg)"},
}

# Signer demographic metadata reported in VDzSL paper
SIGNER_METADATA = {
    "signer0": {"avatar_id": "avatar_0", "gender": "male", "type": "synthetic_3d"},
    "signer1": {"avatar_id": "avatar_1", "gender": "female", "type": "synthetic_3d"},
    "signer2": {"avatar_id": "avatar_2", "gender": "male", "type": "synthetic_3d"},
    "signer3": {"avatar_id": "avatar_3", "gender": "female", "type": "synthetic_3d"},
}


def audit_single_video(file_path: str, dataset_root: str) -> dict:
    """
    Audit an individual video file:
    - Verifies file accessibility and byte size
    - Decodes video frames to check readability and count frames
    - Parses signer, camera angle, and canonical sign label
    """
    rel_path = os.path.relpath(file_path, dataset_root)
    parts = rel_path.replace("\\", "/").split("/")

    signer_id = parts[0] if len(parts) >= 1 else "unknown"
    camera_id = parts[1] if len(parts) >= 2 else "unknown"
    raw_stem, canonical_label = LabelParser.extract_label_from_path(file_path)

    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    record = {
        "rel_path": rel_path,
        "abs_path": os.path.abspath(file_path),
        "raw_label": raw_stem,
        "canonical_label": canonical_label,
        "label_source": "filename",
        "is_numeric_sign": canonical_label.isdigit(),
        "signer_id": signer_id,
        "camera_id": camera_id,
        "file_size_bytes": file_size,
        "is_readable": False,
        "width": 0,
        "height": 0,
        "frame_count": 0,
        "container_fps": 0.0,
        "duration_sec": 0.0,
        "exclusion_reason": None,
    }

    if file_size == 0:
        record["exclusion_reason"] = "zero_byte_file"
        return record

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        record["exclusion_reason"] = "cannot_open_videocapture"
        return record

    container_fps = cap.get(cv2.CAP_PROP_FPS)
    record["container_fps"] = container_fps

    # Read all frames to verify full stream integrity and obtain exact frame count
    frame_count = 0
    w, h = 0, 0
    read_error = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count == 0:
                h, w = frame.shape[:2]
            frame_count += 1
    except Exception as exc:
        read_error = True
        record["exclusion_reason"] = f"read_exception_{str(exc)}"
    finally:
        cap.release()

    if read_error or frame_count == 0:
        if not record["exclusion_reason"]:
            record["exclusion_reason"] = "zero_frames_decoded"
        return record

    record["is_readable"] = True
    record["width"] = w
    record["height"] = h
    record["frame_count"] = frame_count

    # In VDzSL container metadata, container_fps often reports 1000.0 (timebase default).
    # Actual animation FPS is ~27-30 FPS. We calculate duration using 30 FPS nominal.
    effective_fps = 30.0 if (container_fps <= 0 or container_fps > 120) else container_fps
    record["duration_sec"] = round(frame_count / effective_fps, 3)

    return record


def audit_entire_dataset(
    dataset_dir: str,
    max_workers: int = 4,
    sample_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Audit all MP4 files in dataset_dir using multi-threading.
    """
    video_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, f))

    video_files.sort()
    if sample_limit:
        video_files = video_files[:sample_limit]

    print(f"Discovered {len(video_files)} video files in {dataset_dir}")
    print(f"Beginning multi-threaded audit using {max_workers} worker threads...")

    records = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(audit_single_video, path, dataset_dir): path
            for path in video_files
        }

        with tqdm(total=len(video_files), desc="Auditing videos", unit="video") as pbar:
            for future in as_completed(future_to_path):
                rec = future.result()
                records.append(rec)
                pbar.update(1)

    elapsed = time.time() - t0
    print(f"Completed audit of {len(records)} videos in {elapsed:.2f} seconds ({elapsed/len(records)*1000:.1f} ms/video).")

    df = pd.DataFrame(records)
    # Sort deterministically by signer, camera, canonical_label
    df.sort_values(by=["signer_id", "camera_id", "canonical_label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
