"""
Unit tests for feature extraction module.
"""

import os
import numpy as np
import pytest
from src.features.feature_extractor import PixelMotionFeatureExtractor


def test_pixel_motion_feature_extractor():
    extractor = PixelMotionFeatureExtractor(num_frames=8, target_size=(16, 16))

    video_path = r"data\VDzSL\signer0\cam0\1.mp4"
    if not os.path.exists(video_path):
        pytest.skip(f"Dataset video not found at {video_path}")

    features = extractor.extract_from_path(video_path)

    # 16x16 = 256 per image. 4 summaries = 256 * 4 = 1024 features
    assert features.shape == (1024,)
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_pixel_motion_default_dimension():
    extractor = PixelMotionFeatureExtractor(num_frames=16, target_size=(32, 32))

    video_path = r"data\VDzSL\signer0\cam0\1.mp4"
    if not os.path.exists(video_path):
        pytest.skip(f"Dataset video not found at {video_path}")

    features = extractor.extract_from_path(video_path)
    # 32x32 = 1024 per image. 4 summaries = 1024 * 4 = 4096 features
    assert features.shape == (4096,)
    assert features.dtype == np.float32
