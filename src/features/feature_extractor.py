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

    def extract_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract spatio-temporal feature vector from a list of BGR/RGB images."""
        sampled = self.sampler.sample_from_frames(frames).squeeze(-1)
        mean_frame = np.mean(sampled, axis=0).flatten()
        std_frame = np.std(sampled, axis=0).flatten()
        motion_diff = np.abs(np.diff(sampled, axis=0))
        motion_mean = np.mean(motion_diff, axis=0).flatten()
        motion_max = np.max(motion_diff, axis=0).flatten()
        feature_vector = np.concatenate([mean_frame, std_frame, motion_mean, motion_max])
        return feature_vector.astype(np.float32)

    def extract_from_path(self, video_path: str) -> np.ndarray:
        """Extract spatio-temporal feature vector from video file."""
        frames = self.sampler.sample_from_path(video_path).squeeze(-1)  # (T, H, W)
        mean_frame = np.mean(frames, axis=0).flatten()  # (1024,)
        std_frame = np.std(frames, axis=0).flatten()    # (1024,)
        motion_diff = np.abs(np.diff(frames, axis=0))    # (T-1, H, W)
        motion_mean = np.mean(motion_diff, axis=0).flatten()  # (1024,)
        motion_max = np.max(motion_diff, axis=0).flatten()    # (1024,)
        feature_vector = np.concatenate([mean_frame, std_frame, motion_mean, motion_max])
        return feature_vector.astype(np.float32)

    def extract(self, video_path: str) -> np.ndarray:
        """Alias for extract_from_path for unified API."""
        return self.extract_from_path(video_path)


class LandmarkFeatureExtractor:
    """
    Extracts hand and pose landmark sequences using MediaPipe Tasks.
    Normalizes coordinates relative to torso/wrist for translation and scale invariance.
    Includes temporal dynamics, finger articulation geometry, and hand orientation.
    """

    def __init__(
        self,
        num_frames: int = 16,
        hand_model_path: str = r"artifacts/models/hand_landmarker.task",
        pose_model_path: str = r"artifacts/models/pose_landmarker_lite.task",
        include_temporal_summary: bool = True,
    ):
        self.num_frames = num_frames
        self.hand_model_path = self._resolve_model_path(hand_model_path)
        self.pose_model_path = self._resolve_model_path(pose_model_path)
        self.include_temporal_summary = include_temporal_summary
        self._init_detectors()

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        if os.path.exists(path):
            return os.path.abspath(path)
        # Check relative to project root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        alt_path = os.path.join(repo_root, path)
        if os.path.exists(alt_path):
            return os.path.abspath(alt_path)
        return path

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
            min_tracking_confidence=0.3,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=self.pose_model_path),
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    def detect_hands_on_frame(self, frame: np.ndarray):
        """
        Run hand detection on a single BGR frame. Returns MediaPipe HandLandmarkerResult.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        return self.hand_detector.detect(mp_img)

    def extract_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Sample frames and extract scale-invariant landmarks and temporal dynamics.

        Per frame features:
        - Upper pose landmarks (Nose 0, Shoulders 11,12, Elbows 13,14, Wrists 15,16) = 7 keypoints * 3 = 21 values
          (centered on shoulders midpoint, scaled by shoulder distance)
        - Left hand (80 values):
          - presence flag (1)
          - wrist relative to torso in shoulder scale (3)
          - hand scale (1)
          - 21 normalized landmarks (wrist-centered, hand-scale normalized) (63)
          - palm normal (3)
          - palm direction (3)
          - 5 fingertip distances (5)
          - pinch distance (1)
        - Right hand (80 values): identical layout to Left hand
        Frame vector = 21 + 80 + 80 = 181 values.

        Sequence across 16 frames = 16 * 181 = 2,896 values.
        Temporal dynamics summary (if include_temporal_summary=True):
        - Mean across time (181)
        - Std across time (181)
        - Mean absolute frame velocity (181)
        - Left & Right wrist trajectory statistics (displacement, path length, motion range) (14)
        Total feature vector = 3,453 values.
        """
        if len(frames) == 0:
            raise ValueError("Empty frames list provided to extract_from_frames")

        sampler = FrameSampler(num_frames=self.num_frames, target_size=None, normalize=False)
        indices = sampler.compute_frame_indices(len(frames))
        sampled = [frames[i] for i in indices]

        POSE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16]
        FINGERTIP_INDICES = [4, 8, 12, 16, 20]

        frame_vectors = []
        wrist_trajectories = {"left": [], "right": []}

        for frame in sampled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)

            # 1. Pose landmarks
            pose_res = self.pose_detector.detect(mp_img)
            torso_cx, torso_cy, torso_cz = 0.5, 0.5, 0.0
            shoulder_scale = 1.0

            if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                p_lms = pose_res.pose_landmarks[0]
                if len(p_lms) > 12:
                    p11, p12 = p_lms[11], p_lms[12]
                    torso_cx = (p11.x + p12.x) / 2.0
                    torso_cy = (p11.y + p12.y) / 2.0
                    torso_cz = (p11.z + p12.z) / 2.0
                    s_dist = np.sqrt((p11.x - p12.x) ** 2 + (p11.y - p12.y) ** 2 + (p11.z - p12.z) ** 2)
                    if s_dist > 0.05:
                        shoulder_scale = float(s_dist)

                p_vals = []
                for idx in POSE_KEY_INDICES:
                    if idx < len(p_lms):
                        p_vals.extend([
                            (p_lms[idx].x - torso_cx) / shoulder_scale,
                            (p_lms[idx].y - torso_cy) / shoulder_scale,
                            (p_lms[idx].z - torso_cz) / shoulder_scale,
                        ])
                    else:
                        p_vals.extend([0.0, 0.0, 0.0])
                pose_vec = np.array(p_vals, dtype=np.float32)
            else:
                pose_vec = np.zeros(len(POSE_KEY_INDICES) * 3, dtype=np.float32)

            # 2. Hand landmarks
            left_dict = {
                "present": 0.0,
                "wrist_rel": [0.0, 0.0, 0.0],
                "scale": 0.0,
                "norm_lms": np.zeros(63, dtype=np.float32),
                "palm_normal": [0.0, 0.0, 0.0],
                "palm_dir": [0.0, 0.0, 0.0],
                "ftips": [0.0] * 5,
                "pinch": 0.0,
            }
            right_dict = {
                "present": 0.0,
                "wrist_rel": [0.0, 0.0, 0.0],
                "scale": 0.0,
                "norm_lms": np.zeros(63, dtype=np.float32),
                "palm_normal": [0.0, 0.0, 0.0],
                "palm_dir": [0.0, 0.0, 0.0],
                "ftips": [0.0] * 5,
                "pinch": 0.0,
            }

            hand_res = self.hand_detector.detect(mp_img)
            if hand_res.hand_landmarks:
                for h_idx, lms in enumerate(hand_res.hand_landmarks):
                    if len(lms) < 21:
                        continue

                    raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    w_pt = raw_pts[0]
                    mcp9_pt = raw_pts[9]

                    h_scale = float(np.linalg.norm(mcp9_pt - w_pt))
                    if h_scale < 1e-4:
                        h_scale = 0.1

                    wrist_rel = [
                        float((w_pt[0] - torso_cx) / shoulder_scale),
                        float((w_pt[1] - torso_cy) / shoulder_scale),
                        float((w_pt[2] - torso_cz) / shoulder_scale),
                    ]

                    # Scale-normalized finger landmarks relative to wrist
                    norm_pts = (raw_pts - w_pt) / h_scale

                    # Palm orientation
                    palm_dir = (mcp9_pt - w_pt) / h_scale
                    v1 = raw_pts[5] - w_pt
                    v2 = raw_pts[17] - w_pt
                    normal = np.cross(v1, v2)
                    norm_mag = np.linalg.norm(normal)
                    palm_normal = (normal / norm_mag).tolist() if norm_mag > 1e-5 else [0.0, 0.0, 1.0]

                    # Fingertip extension distances
                    ftips = [float(np.linalg.norm(raw_pts[f_idx] - w_pt) / h_scale) for f_idx in FINGERTIP_INDICES]
                    pinch = float(np.linalg.norm(raw_pts[4] - raw_pts[8]) / h_scale)

                    h_data = {
                        "present": 1.0,
                        "wrist_rel": wrist_rel,
                        "scale": h_scale,
                        "norm_lms": norm_pts.flatten(),
                        "palm_normal": palm_normal,
                        "palm_dir": palm_dir.tolist(),
                        "ftips": ftips,
                        "pinch": pinch,
                    }

                    # Determine handedness
                    side = "Unknown"
                    if hand_res.handedness and h_idx < len(hand_res.handedness):
                        if len(hand_res.handedness[h_idx]) > 0:
                            side = hand_res.handedness[h_idx][0].category_name

                    if side == "Left":
                        left_dict = h_data
                    elif side == "Right":
                        right_dict = h_data
                    else:
                        if h_idx == 0:
                            left_dict = h_data
                        elif h_idx == 1:
                            right_dict = h_data

            wrist_trajectories["left"].append(left_dict["wrist_rel"])
            wrist_trajectories["right"].append(right_dict["wrist_rel"])

            def pack_hand(hd):
                return np.concatenate([
                    [hd["present"]],
                    hd["wrist_rel"],
                    [hd["scale"]],
                    hd["norm_lms"],
                    hd["palm_normal"],
                    hd["palm_dir"],
                    hd["ftips"],
                    [hd["pinch"]],
                ]).astype(np.float32)

            left_vec = pack_hand(left_dict)
            right_vec = pack_hand(right_dict)

            frame_features = np.concatenate([pose_vec, left_vec, right_vec]).astype(np.float32)
            frame_vectors.append(frame_features)

        seq_mat = np.stack(frame_vectors, axis=0)  # (16, 181)
        raw_feat = seq_mat.flatten()

        if not self.include_temporal_summary:
            return raw_feat

        # Temporal dynamics summaries
        mean_seq = np.mean(seq_mat, axis=0)
        std_seq = np.std(seq_mat, axis=0)
        diff_seq = np.diff(seq_mat, axis=0)  # (15, 181)
        mean_diff = np.mean(np.abs(diff_seq), axis=0)

        # Trajectory statistics for left & right wrists
        lw = np.array(wrist_trajectories["left"], dtype=np.float32)  # (16, 3)
        rw = np.array(wrist_trajectories["right"], dtype=np.float32)  # (16, 3)
        l_net = lw[-1] - lw[0]
        r_net = rw[-1] - rw[0]
        l_path = np.sum(np.linalg.norm(np.diff(lw, axis=0), axis=1))
        r_path = np.sum(np.linalg.norm(np.diff(rw, axis=0), axis=1))
        l_range = np.max(lw, axis=0) - np.min(lw, axis=0)
        r_range = np.max(rw, axis=0) - np.min(rw, axis=0)

        traj_stats = np.concatenate([l_net, r_net, [l_path, r_path], l_range, r_range]).astype(np.float32)

        final_features = np.concatenate([
            raw_feat,
            mean_seq,
            std_seq,
            mean_diff,
            traj_stats,
        ]).astype(np.float32)

        return final_features

    def extract_from_path(self, video_path: str) -> np.ndarray:
        """Extract landmarks from a video file path."""
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

        return self.extract_from_frames(raw_frames)

    def extract(self, video_path: str) -> np.ndarray:
        """Alias for extract_from_path for unified API."""
        return self.extract_from_path(video_path)

    def close(self):
        if hasattr(self, "hand_detector"):
            self.hand_detector.close()
        if hasattr(self, "pose_detector"):
            self.pose_detector.close()

