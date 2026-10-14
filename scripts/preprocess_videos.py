"""
Preprocess videos and cache extracted feature matrices to disk.

Usage:
    python scripts/preprocess_videos.py [--manifest data/manifests/split_mvp_stratified.csv] [--mode pixel|landmark] [--output data/processed/features_mvp_pixel.npz]
"""

import argparse
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


def preprocess_dataset(
    manifest_csv: str,
    output_npz: str,
    mode: str = "pixel",
    labels_json: str = r"artifacts\label_mappings\mvp_classes.json",
):
    df = pd.read_csv(manifest_csv, encoding="utf-8-sig")
    label_parser = LabelParser.load(labels_json)

    print(f"Loaded manifest with {len(df)} videos across {df['canonical_label'].nunique()} classes.")
    print(f"Initializing feature extractor (mode: '{mode}')...")

    if mode == "pixel":
        extractor = PixelMotionFeatureExtractor(num_frames=16, target_size=(32, 32))
    elif mode == "landmark":
        extractor = LandmarkFeatureExtractor(num_frames=16)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    X_list = []
    y_list = []
    rel_paths = []
    signers = []
    cameras = []
    splits = []
    valid_indices = []

    t0 = time.time()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {mode} features"):
        video_path = row["abs_path"] if "abs_path" in row and os.path.exists(str(row["abs_path"])) else os.path.join("data", "VDzSL", row["rel_path"])

        try:
            feat = extractor.extract_from_path(video_path)
            label_id = label_parser.encode(row["canonical_label"])

            X_list.append(feat)
            y_list.append(label_id)
            rel_paths.append(row["rel_path"])
            signers.append(row["signer_id"])
            cameras.append(row["camera_id"])
            splits.append(row.get("split", "unknown"))
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    if mode == "landmark":
        extractor.close()

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
        default=r"data\processed\features_mvp_pixel.npz",
        help="Path to output NPZ file",
    )
    parser.add_argument(
        "--mode",
        choices=["pixel", "landmark"],
        default="pixel",
        help="Feature extraction mode",
    )
    parser.add_argument(
        "--labels-json",
        default=r"artifacts\label_mappings\mvp_classes.json",
        help="Path to MVP classes JSON",
    )
    args = parser.parse_args()

    preprocess_dataset(args.manifest, args.output, mode=args.mode, labels_json=args.labels_json)


if __name__ == "__main__":
    main()
