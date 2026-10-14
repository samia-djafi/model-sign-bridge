"""
Unit tests for data splitting module.
"""

import pandas as pd
import pytest
from src.data.splits import (
    create_avatar_generalization_split,
    create_camera_generalization_split,
    create_stratified_split,
    filter_subset,
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset with 4 signers, 5 cameras, 4 classes."""
    data = []
    classes = ["ألم", "طبيب أسنان", "1", "ماء"]
    signers = ["signer0", "signer1", "signer2", "signer3"]
    cameras = ["cam0", "cam1", "cam2", "cam3", "cam4"]

    for c in classes:
        for s in signers:
            for cam in cameras:
                data.append({
                    "canonical_label": c,
                    "signer_id": s,
                    "camera_id": cam,
                    "rel_path": f"{s}/{cam}/{c}.mp4"
                })
    return pd.DataFrame(data)


def test_avatar_generalization_split(mock_dataset):
    df_split = create_avatar_generalization_split(mock_dataset, test_signer="signer3", val_signer="signer2")

    # Verify no test samples come from train signers
    test_signers = set(df_split[df_split["split"] == "test"]["signer_id"])
    train_signers = set(df_split[df_split["split"] == "train"]["signer_id"])
    val_signers = set(df_split[df_split["split"] == "val"]["signer_id"])

    assert test_signers == {"signer3"}
    assert val_signers == {"signer2"}
    assert train_signers == {"signer0", "signer1"}
    assert test_signers.isdisjoint(train_signers)


def test_camera_generalization_split(mock_dataset):
    df_split = create_camera_generalization_split(mock_dataset, test_cam="cam4", val_cam="cam3")

    test_cams = set(df_split[df_split["split"] == "test"]["camera_id"])
    train_cams = set(df_split[df_split["split"] == "train"]["camera_id"])
    val_cams = set(df_split[df_split["split"] == "val"]["camera_id"])

    assert test_cams == {"cam4"}
    assert val_cams == {"cam3"}
    assert train_cams == {"cam0", "cam1", "cam2"}
    assert test_cams.isdisjoint(train_cams)


def test_stratified_split(mock_dataset):
    df_split = create_stratified_split(mock_dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)

    assert set(df_split["split"]) == {"train", "val", "test"}
    # Verify every class is present in train, val, and test
    for split_name in ["train", "val", "test"]:
        split_classes = set(df_split[df_split["split"] == split_name]["canonical_label"])
        assert split_classes == {"ألم", "طبيب أسنان", "1", "ماء"}


def test_filter_subset(mock_dataset):
    subset = filter_subset(mock_dataset, target_labels=["ألم", "ماء"])
    assert set(subset["canonical_label"]) == {"ألم", "ماء"}
    assert len(subset) == 2 * 4 * 5  # 40 videos
