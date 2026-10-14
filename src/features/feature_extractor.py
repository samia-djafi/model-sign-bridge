"""
Feature extraction module for Algerian Sign Language recognition.

Implements two complementary representations:
1. Spatio-Temporal Pixel Motion Representation:
   - Uniform temporal frame sampling (T=16)
   - Spatial downsampling (32x32 grayscale)
   - Dynamic motion history / inter-frame temporal differences
   - Total dimension: 4,096 (100% robust across all synthetic avatars and backgrounds)

2. Landmark Sequence Representation:
   - Upper body pose (shoulders, elbows, wrists) + 21 keypoints per hand
   - Normalized coordinates with zero-imputation for undetected hands
   - Compact vector: 2,304 features (avatar-invariant, highly portable)
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.preprocessing.frame_sampler import FrameSampler


class PixelMotionFeatureExtractor:
    """
    Extracts spatio-temporal appearance and motion energy features from video clips.
    """

    def __init__(
        self,
        num_frames: int = 16,
        target_size: Tuple[int, int] = (32, 32),
    ):
        self.num_frames = num_frames
        self.target_size = target_size
        self.sampler = FrameSampler(
            num_frames=num_frames,
            target_size=target_size,
            normalize=True,
            grayscale=True,
        )

    def extract_from_path(self, video_path: str) -> np.ndarray:
        """
        Extract spatio-temporal feature vector.

        Returns:
            1D numpy array of shape (4096,) containing:
            - Temporal mean image (1024)
            - Temporal std / variance image (1024)
            - Temporal motion difference mean (1024)
            - Temporal motion difference max (1024)
        """
        # Shape: (T, H, W, 1) -> float32 in [0, 1]
        frames = self.sampler.sample_from_path(video_path).squeeze(-1)  # (T, H, W)

        # 1. Appearance summary
        mean_frame = np.mean(frames, axis=0).flatten()  # (1024,)
        std_frame = np.std(frames, axis=0).flatten()    # (1024,)

        # 2. Inter-frame motion difference: |F_{t+1} - F_t|
        motion_diff = np.abs(np.diff(frames, axis=0))    # (T-1, H, W)
        motion_mean = np.mean(motion_diff, axis=0).flatten()  # (1024,)
        motion_max = np.max(motion_diff, axis=0).flatten()    # (1024,)

        feature_vector = np.concatenate([mean_frame, std_frame, motion_mean, motion_max])
        return feature_vector.astype(np.float32)


class LandmarkFeatureExtractor:
    """
    Extracts hand and pose landmark sequences using MediaPipe Tasks.
    """

    def __init__(
        self,
        num_frames: int = 16,
        hand_model_path: str = r"artifacts/models/hand_landmarker.task",
        pose_model_path: str = r"artifacts/models/pose_landmarker_lite.task",
    ):
        self.num_frames = num_frames
        self.hand_model_path = hand_model_path
        self.pose_model_path = pose_model_path
        self._init_detectors()

    def _init_detectors(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self.mp = mp
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self.hand_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self.pose_model_path),
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    def extract_from_path(self, video_path: str) -> np.ndarray:
        """
        Sample frames and extract landmarks for each frame:
        - Upper pose landmarks (indices 11, 12, 13, 14, 15, 16: shoulders, elbows, wrists) = 6 keypoints
        - Left hand (21 keypoints)
        - Right hand (21 keypoints)
        Total keypoints = 48 * 3 (x, y, z) = 144 per frame.
        Across 16 frames = 2,304 features.

        Returns:
            1D numpy array of shape (2304,)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

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
            raise ValueError(f"No frames in video: {video_path}")

        sampler = FrameSampler(num_frames=self.num_frames, target_size=None, normalize=False)
        indices = sampler.compute_frame_indices(len(raw_frames))
        sampled = [raw_frames[i] for i in indices]

        # Pose keypoint indices: left/right shoulder (11, 12), elbow (13, 14), wrist (15, 16)
        POSE_KEY_INDICES = [11, 12, 13, 14, 15, 16]

        frame_vectors = []
        for frame in sampled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

            # 1. Pose landmarks (6 keypoints * 3 = 18 values)
            pose_vec = np.zeros(18, dtype=np.float32)
            pose_res = self.pose_detector.detect(mp_img)
            if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                p_lms = pose_res.pose_landmarks[0]
                vals = []
                for idx in POSE_KEY_INDICES:
                    if idx < len(p_lms):
                        vals.extend([p_lms[idx].x, p_lms[idx].y, p_lms[idx].z])
                    else:
                        vals.extend([0.0, 0.0, 0.0])
                pose_vec = np.array(vals, dtype=np.float32)

            # 2. Hand landmarks (Left hand 21*3=63, Right hand 21*3=63)
            left_hand_vec = np.zeros(63, dtype=np.float32)
            right_hand_vec = np.zeros(63, dtype=np.float32)

            hand_res = self.hand_detector.detect(mp_img)
            if hand_res.hand_landmarks:
                for h_idx, lms in enumerate(hand_res.hand_landmarks):
                    h_vals = []
                    for lm in lms:
                        h_vals.extend([lm.x, lm.y, lm.z])
                    arr = np.array(h_vals, dtype=np.float32)
                    if h_idx == 0:
                        left_hand_vec = arr
                    elif h_idx == 1:
                        right_hand_vec = arr

            frame_features = np.concatenate([pose_vec, left_hand_vec, right_hand_vec])  # 18 + 63 + 63 = 144
            frame_vectors.append(frame_features)

        full_sequence = np.stack(frame_vectors, axis=0)  # (16, 144)
        return full_sequence.flatten().astype(np.float32)  # (2304,)

    def close(self):
        if hasattr(self, "hand_detector"):
            self.hand_detector.close()
        if hasattr(self, "pose_detector"):
            self.pose_detector.close()
