"""
Video frame sampling and resizing module for Algerian Sign Language models.

Provides deterministic temporal sampling, uniform frame index selection,
padding for short clips, and consistent spatial normalization.
"""

import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


class FrameSampler:
    """
    Sample a fixed number of frames uniformly across a video duration,
    resizing and normalizing them consistently for model consumption.
    """

    def __init__(
        self,
        num_frames: int = 16,
        target_size: Optional[Tuple[int, int]] = (64, 64),
        normalize: bool = True,
        grayscale: bool = False,
    ):
        """
        Args:
            num_frames: Number of temporal frames to sample (T).
            target_size: Target (width, height) to resize frames to. If None, keeps original resolution.
            normalize: If True, scales pixel values from [0, 255] to [0.0, 1.0].
            grayscale: If True, converts 3-channel RGB/BGR to 1-channel grayscale.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        self.num_frames = num_frames
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale

    def compute_frame_indices(self, total_frames: int) -> np.ndarray:
        """
        Compute deterministic uniform frame indices.

        Args:
            total_frames: Total number of frames in the video.

        Returns:
            1D numpy array of integer indices of length self.num_frames.
        """
        if total_frames <= 0:
            raise ValueError(f"total_frames must be > 0, got {total_frames}")

        if total_frames == 1:
            return np.zeros(self.num_frames, dtype=int)

        if total_frames >= self.num_frames:
            # Uniform linspace between first and last frame
            indices = np.linspace(0, total_frames - 1, num=self.num_frames, dtype=int)
        else:
            # Short clip: pad with repeated last frame
            indices = np.pad(
                np.arange(total_frames),
                (0, self.num_frames - total_frames),
                mode="edge",
            )

        return indices

    def sample_from_path(self, video_path: str) -> np.ndarray:
        """
        Open a video file, read all frames sequentially, sample uniformly,
        and apply spatial transformations.

        Args:
            video_path: Path to MP4 file.

        Returns:
            Numpy array of shape:
                (T, H, W, C) if grayscale=False (C=3, BGR or RGB depending on cv2)
                (T, H, W, 1) if grayscale=True
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file via OpenCV: {video_path}")

        raw_frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
        finally:
            cap.release()

        if len(raw_frames) == 0:
            raise ValueError(f"Video contained 0 decodable frames: {video_path}")

        indices = self.compute_frame_indices(len(raw_frames))
        sampled = [raw_frames[idx] for idx in indices]

        processed_frames = []
        for f in sampled:
            # Optional Grayscale conversion
            if self.grayscale:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = np.expand_dims(f, axis=-1)

            # Spatial resize
            if self.target_size is not None:
                tw, th = self.target_size
                f = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
                if self.grayscale and len(f.shape) == 2:
                    f = np.expand_dims(f, axis=-1)

            processed_frames.append(f)

        tensor = np.stack(processed_frames, axis=0)  # (T, H, W, C)

        # Normalization
        if self.normalize:
            tensor = tensor.astype(np.float32) / 255.0

        return tensor
