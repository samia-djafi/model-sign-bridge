"""
Build data split manifests for Algerian Sign Language models.

Generates:
1. Stratified In-Domain Split (70% train, 15% val, 15% test)
2. Avatar Generalization Split (Train: signer0, signer1; Val: signer2; Test: signer3)
3. Camera Viewpoint Generalization Split (Train: cam0..cam2; Val: cam3; Test: cam4)
"""

import argparse
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.data.label_parser import LabelParser
from src.data.splits import (
    create_avatar_generalization_split,
    create_camera_generalization_split,
    create_stratified_split,
    filter_subset,
)


def main():
    parser = argparse.ArgumentParser(description="Build split manifests for VDzSL.")
    parser.add_argument(
        "--inventory-csv",
        default=r"data\manifests\dataset_inventory.csv",
        help="Path to full inventory CSV",
    )
    parser.add_argument(
        "--vocab-csv",
        default=r"data\manifests\medical_vocabulary.csv",
        help="Path to medical vocabulary CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=r"data\manifests",
        help="Directory to save split manifests",
    )
    args = parser.parse_args()

    df_inv = pd.read_csv(args.inventory_csv, encoding="utf-8-sig")
    df_vocab = pd.read_csv(args.vocab_csv, encoding="utf-8-sig")

    mvp_labels = df_vocab[df_vocab["suitable_for_mvp"]]["original_label"].tolist()
    print(f"Loaded {len(mvp_labels)} MVP healthcare/pharmacy classes.")

    # Filter to MVP subset (32 classes x 20 videos = 640 videos)
    df_mvp = filter_subset(df_inv, mvp_labels)
    print(f"MVP dataset size: {len(df_mvp)} videos ({len(df_mvp[df_mvp['is_readable']])} readable).")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Stratified Split (70% train, 15% val, 15% test)
    df_strat = create_stratified_split(df_mvp, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    strat_csv = os.path.join(args.output_dir, "split_mvp_stratified.csv")
    df_strat.to_csv(strat_csv, index=False, encoding="utf-8-sig")
    print(f"Saved stratified split to {strat_csv}:")
    print(f"  Train: {len(df_strat[df_strat['split'] == 'train'])}, Val: {len(df_strat[df_strat['split'] == 'val'])}, Test: {len(df_strat[df_strat['split'] == 'test'])}")

    # 2. Avatar Generalization Split (Test: signer3, Val: signer2, Train: signer0, signer1)
    df_avatar = create_avatar_generalization_split(df_mvp, test_signer="signer3", val_signer="signer2")
    avatar_csv = os.path.join(args.output_dir, "split_mvp_avatar_heldout.csv")
    df_avatar.to_csv(avatar_csv, index=False, encoding="utf-8-sig")
    print(f"Saved avatar generalization split to {avatar_csv}:")
    print(f"  Train: {len(df_avatar[df_avatar['split'] == 'train'])}, Val: {len(df_avatar[df_avatar['split'] == 'val'])}, Test: {len(df_avatar[df_avatar['split'] == 'test'])}")

    # 3. Camera Generalization Split (Test: cam4, Val: cam3, Train: cam0..cam2)
    df_cam = create_camera_generalization_split(df_mvp, test_cam="cam4", val_cam="cam3")
    cam_csv = os.path.join(args.output_dir, "split_mvp_camera_heldout.csv")
    df_cam.to_csv(cam_csv, index=False, encoding="utf-8-sig")
    print(f"Saved camera generalization split to {cam_csv}:")
    print(f"  Train: {len(df_cam[df_cam['split'] == 'train'])}, Val: {len(df_cam[df_cam['split'] == 'val'])}, Test: {len(df_cam[df_cam['split'] == 'test'])}")

    # 4. Save canonical label parser mapping specifically for MVP
    mvp_parser = LabelParser(mvp_labels)
    mvp_labels_json = r"artifacts\label_mappings\mvp_classes.json"
    os.makedirs(os.path.dirname(mvp_labels_json), exist_ok=True)
    mvp_parser.save(mvp_labels_json)
    print(f"Saved MVP classes JSON ({mvp_parser.num_classes} classes) to {mvp_labels_json}")


if __name__ == "__main__":
    main()
