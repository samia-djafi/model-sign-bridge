"""
Preprocess videos and cache extracted feature matrices to disk.

Usage:
    # Landmark mode (multiprocessed):
    python scripts/preprocess_videos.py --mode landmark --workers 4 --output data/processed/features_mvp_landmark.npz

    # Pixel mode:
    python scripts/preprocess_videos.py --mode pixel --output data/processed/features_mvp_pixel.npz
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.data.label_parser import LabelParser
from src.features.feature_extractor import LandmarkFeatureExtractor, PixelMotionFeatureExtractor


_worker_extractor = None


def _init_worker(mode: str, num_frames: int):
    global _worker_extractor
    if mode == "landmark":
        _worker_extractor = LandmarkFeatureExtractor(num_frames=num_frames)
    elif mode == "pixel":
        _worker_extractor = PixelMotionFeatureExtractor(num_frames=num_frames)


def _extract_worker(video_path: str):
    global _worker_extractor
    try:
        return _worker_extractor.extract_from_path(video_path), None
    except Exception as e:
        return None, str(e)


def preprocess_dataset(
    manifest_csv: str,
    output_npz: str,
    mode: str = "pixel",
    labels_json: str = r"artifacts\label_mappings\mvp_classes.json",
    workers: int = 1,
    num_frames: int = 16,
):
    df = pd.read_csv(manifest_csv, encoding="utf-8-sig")
    label_parser = LabelParser.load(labels_json)

    print(f"Loaded manifest with {len(df)} videos across {df['canonical_label'].nunique()} classes.")
    print(f"Feature extraction mode: '{mode}' | Workers: {workers} | Frames: {num_frames}")

    # Prepare video paths
    video_paths = []
    for _, row in df.iterrows():
        if "abs_path" in row and os.path.exists(str(row["abs_path"])):
            vpath = str(row["abs_path"])
        else:
            vpath = os.path.join("data", "VDzSL", row["rel_path"])
        video_paths.append(vpath)

    t0 = time.time()
    feats = []

    if workers > 1:
        print(f"Starting ProcessPoolExecutor with {workers} worker processes...")
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mode, num_frames)) as executor:
            for feat, err in tqdm(executor.map(_extract_worker, video_paths), total=len(video_paths), desc=f"Extracting {mode}"):
                if err:
                    print(f"\nWarning: Extraction error: {err}")
                    feats.append(None)
                else:
                    feats.append(feat)
    else:
        if mode == "pixel":
            extractor = PixelMotionFeatureExtractor(num_frames=num_frames, target_size=(32, 32))
        else:
            extractor = LandmarkFeatureExtractor(num_frames=num_frames)

        for vpath in tqdm(video_paths, total=len(video_paths), desc=f"Extracting {mode}"):
            try:
                feat = extractor.extract_from_path(vpath)
                feats.append(feat)
            except Exception as e:
                print(f"\nWarning: Error on {vpath}: {e}")
                feats.append(None)

        if hasattr(extractor, "close"):
            extractor.close()

    # Filter out failures
    X_list = []
    y_list = []
    rel_paths = []
    signers = []
    cameras = []
    splits = []

    for idx, (feat, (_, row)) in enumerate(zip(feats, df.iterrows())):
        if feat is not None:
            label_id = label_parser.encode(row["canonical_label"])
            X_list.append(feat)
            y_list.append(label_id)
            rel_paths.append(row["rel_path"])
            signers.append(row["signer_id"])
            cameras.append(row["camera_id"])
            splits.append(row.get("split", "unknown"))

    elapsed = time.time() - t0
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        rel_paths=np.array(rel_paths),
        signers=np.array(signers),
        cameras=np.array(cameras),
        splits=np.array(splits),
        mode=mode,
        feature_dim=X.shape[1],
        num_samples=X.shape[0],
    )

    file_size_mb = os.path.getsize(output_npz) / (1024 * 1024)
    print(f"\nFeature extraction complete in {elapsed:.2f}s ({elapsed/len(X)*1000:.1f} ms/video)!")
    print(f"Feature matrix shape : {X.shape}")
    print(f"Labels array shape   : {y.shape}")
    print(f"Saved compressed NPZ : {output_npz} ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos and cache features.")
    parser.add_argument(
        "--manifest",
        default=r"data\manifests\split_mvp_stratified.csv",
        help="Path to dataset split manifest",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output NPZ file (defaults to data/processed/features_mvp_<mode>.npz)",
    )
    parser.add_argument(
        "--mode",
        choices=["pixel", "landmark"],
        default="landmark",
        help="Feature extraction mode",
    )
    parser.add_argument(
        "--labels-json",
        default=r"artifacts\label_mappings\mvp_classes.json",
        help="Path to MVP classes JSON",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of uniformly sampled frames per video",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("data", "processed", f"features_mvp_{args.mode}.npz")

    preprocess_dataset(
        args.manifest,
        args.output,
        mode=args.mode,
        labels_json=args.labels_json,
        workers=args.workers,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
