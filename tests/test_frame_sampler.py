"""
Unit tests for FrameSampler.
"""

import os
import numpy as np
import pytest
from src.preprocessing.frame_sampler import FrameSampler


def test_frame_sampler_initialization():
    sampler = FrameSampler(num_frames=16, target_size=(64, 64))
    assert sampler.num_frames == 16
    assert sampler.target_size == (64, 64)

    with pytest.raises(ValueError):
        FrameSampler(num_frames=0)

    with pytest.raises(ValueError):
        FrameSampler(num_frames=-5)


def test_compute_frame_indices():
    sampler = FrameSampler(num_frames=8)

    # 1. Standard video with 30 frames -> 8 uniformly spaced indices
    idx_30 = sampler.compute_frame_indices(30)
    assert len(idx_30) == 8
    assert idx_30[0] == 0
    assert idx_30[-1] == 29
    assert np.all(np.diff(idx_30) >= 0)  # Monotonically increasing

    # 2. Short video with 4 frames -> 8 padded indices
    idx_4 = sampler.compute_frame_indices(4)
    assert len(idx_4) == 8
    assert list(idx_4) == [0, 1, 2, 3, 3, 3, 3, 3]

    # 3. Exact match: 8 frames -> [0, 1, 2, 3, 4, 5, 6, 7]
    idx_8 = sampler.compute_frame_indices(8)
    assert list(idx_8) == list(range(8))

    # 4. Single frame video -> all zeros
    idx_1 = sampler.compute_frame_indices(1)
    assert len(idx_1) == 8
    assert all(i == 0 for i in idx_1)

    # 5. Invalid frame count
    with pytest.raises(ValueError):
        sampler.compute_frame_indices(0)


def test_sample_missing_file():
    sampler = FrameSampler(num_frames=8)
    with pytest.raises(FileNotFoundError):
        sampler.sample_from_path("non_existent_file.mp4")


def test_sample_real_video():
    """Test sampling on an actual VDzSL video."""
    video_path = r"data\VDzSL\signer0\cam0\1.mp4"
    if not os.path.exists(video_path):
        pytest.skip(f"Dataset video not found at {video_path}")

    # RGB Sampling
    sampler_rgb = FrameSampler(num_frames=16, target_size=(64, 64), normalize=True, grayscale=False)
    tensor_rgb = sampler_rgb.sample_from_path(video_path)

    assert tensor_rgb.shape == (16, 64, 64, 3)
    assert tensor_rgb.dtype == np.float32
    assert tensor_rgb.min() >= 0.0
    assert tensor_rgb.max() <= 1.0

    # Grayscale Sampling
    sampler_gray = FrameSampler(num_frames=8, target_size=(32, 32), normalize=True, grayscale=True)
    tensor_gray = sampler_gray.sample_from_path(video_path)

    assert tensor_gray.shape == (8, 32, 32, 1)
    assert tensor_gray.dtype == np.float32
