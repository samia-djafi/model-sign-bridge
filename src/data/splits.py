"""
Dataset splitting module for Algerian Sign Language models.

Provides leakage-aware splitting strategies:
1. Avatar Generalization Split (Unseen Signer / Avatar-held-out):
   - Train: signer0, signer1, signer2
   - Test: signer3 (strictly unseen avatar)
2. Camera Viewpoint Generalization Split (Unseen Camera Angle):
   - Train: cam0, cam1, cam2, cam3
   - Test: cam4 (strictly unseen camera angle)
3. Stratified In-Distribution Split (Standard random train/val/test)
"""

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_avatar_generalization_split(
    df: pd.DataFrame,
    test_signer: str = "signer3",
    val_signer: Optional[str] = "signer2",
) -> pd.DataFrame:
    """
    Split data ensuring the test set contains an entirely unseen avatar.
    If val_signer is specified, holds out that avatar for validation.
    """
    df_split = df.copy()
    df_split["split"] = "train"

    df_split.loc[df_split["signer_id"] == test_signer, "split"] = "test"
    if val_signer:
        df_split.loc[df_split["signer_id"] == val_signer, "split"] = "val"

    return df_split


def create_camera_generalization_split(
    df: pd.DataFrame,
    test_cam: str = "cam4",
    val_cam: Optional[str] = "cam3",
) -> pd.DataFrame:
    """
    Split data ensuring the test set contains an entirely unseen camera viewpoint.
    """
    df_split = df.copy()
    df_split["split"] = "train"

    df_split.loc[df_split["camera_id"] == test_cam, "split"] = "test"
    if val_cam:
        df_split.loc[df_split["camera_id"] == val_cam, "split"] = "val"

    return df_split


def create_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a stratified random train/val/test split across all classes.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5

    df_split = df.copy().reset_index(drop=True)
    df_split["split"] = "unassigned"

    # Step 1: Split into train+val vs test
    test_size = test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(df_split, df_split["canonical_label"]))

    df_split.loc[test_idx, "split"] = "test"

    # Step 2: Split train_val into train and val
    val_scaled_ratio = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_scaled_ratio, random_state=random_state)
    train_sub_idx, val_sub_idx = next(sss2.split(df_split.loc[train_val_idx], df_split.loc[train_val_idx, "canonical_label"]))

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    df_split.loc[train_idx, "split"] = "train"
    df_split.loc[val_idx, "split"] = "val"

    return df_split


def filter_subset(
    inventory_df: pd.DataFrame,
    target_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter inventory DataFrame to a target subset of classes (e.g. Medical MVP).
    """
    if target_labels is None:
        return inventory_df.copy()

    subset = inventory_df[inventory_df["canonical_label"].isin(target_labels)].copy()
    subset.reset_index(drop=True, inplace=True)
    return subset
