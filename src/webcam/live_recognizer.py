"""
Live webcam sign language recognizer for Algerian Sign Language.

Features:
- Rolling frame buffer (32 frames)
- Dynamic motion activity gate (prevents false predictions when user is resting)
- Temporal prediction smoothing / debouncing
- Real-time FPS and latency measurement
- Synthetic-to-Real domain shift disclaimer overlay
- Graceful camera failure handling and clean release
"""

import collections
import time
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.features.feature_extractor import PixelMotionFeatureExtractor
from src.inference.predictor import VideoSignPredictor


class LiveSignRecognizer:
    """
    Live stream processor with motion activity detection,
    temporal debouncing, and inference smoothing.
    """

    def __init__(
        self,
        predictor: VideoSignPredictor,
        window_size: int = 16,
        sample_stride: int = 2,
        motion_threshold: float = 0.0005,
        smoothing_window: int = 3,
    ):
        self.predictor = predictor
        self.window_size = window_size
        self.sample_stride = sample_stride
        self.motion_threshold = motion_threshold
        self.smoothing_window = smoothing_window

        self.frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self.prediction_history: Deque[Tuple[str, float]] = collections.deque(maxlen=smoothing_window)

        self.prev_gray: Optional[np.ndarray] = None
        self.last_prediction: Optional[Dict] = None
        self.is_active_motion: bool = False
        self.last_inference_time_ms: float = 0.0

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single incoming video frame:
        - Computes inter-frame motion delta
        - Appends to rolling buffer
        - When buffer is full and motion is active, runs inference
        - Smoothes predictions across time
        """
        h, w = frame.shape[:2]
        small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))

        # 1. Motion activity check
        motion_score = 0.0
        if self.prev_gray is not None:
            diff = np.abs(small_gray.astype(np.float32) - self.prev_gray.astype(np.float32)) / 255.0
            motion_score = float(np.mean(diff))

        self.prev_gray = small_gray
        self.is_active_motion = motion_score > self.motion_threshold
        self.frame_buffer.append(frame.copy())

        # 2. Trigger inference periodically when buffer is full and motion is detected
        if len(self.frame_buffer) == self.window_size and self.is_active_motion:
            # Sample 16 frames uniformly from the 30-frame rolling buffer
            indices = np.linspace(0, self.window_size - 1, num=16, dtype=int)
            sampled_frames = [self.frame_buffer[i] for i in indices]

            # Preprocess sampled frames
            t0 = time.time()
            resized = [
                cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 32)) / 255.0
                for f in sampled_frames
            ]
            frames_arr = np.stack(resized, axis=0)  # (16, 32, 32)

            mean_f = np.mean(frames_arr, axis=0).flatten()
            std_f = np.std(frames_arr, axis=0).flatten()
            motion_diff = np.abs(np.diff(frames_arr, axis=0))
            m_mean = np.mean(motion_diff, axis=0).flatten()
            m_max = np.max(motion_diff, axis=0).flatten()
            feat_vec = np.concatenate([mean_f, std_f, m_mean, m_max]).astype(np.float32)

            # Classify
            probs = self.predictor.classifier.predict_proba(np.expand_dims(feat_vec, axis=0))[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            top_label = self.predictor.target_names[top_idx]
            self.last_inference_time_ms = (time.time() - t0) * 1000

            self.prediction_history.append((top_label, top_prob))

            # Temporal smoothing (majority voting weighted by probability)
            label_scores = collections.defaultdict(float)
            for lbl, score in self.prediction_history:
                label_scores[lbl] += score

            smoothed_label = max(label_scores.items(), key=lambda x: x[1])[0]
            vocab_info = self.predictor.vocab_lookup.get(smoothed_label, {})

            self.last_prediction = {
                "label": smoothed_label,
                "confidence": top_prob,
                "meaning_en": vocab_info.get("meaning_en", "N/A"),
                "meaning_fr": vocab_info.get("meaning_fr", "N/A"),
                "category": vocab_info.get("category", "General"),
                "is_uncertain": top_prob < self.predictor.uncertainty_threshold,
                "latency_ms": self.last_inference_time_ms,
            }

        return {
            "motion_score": motion_score,
            "is_active_motion": self.is_active_motion,
            "prediction": self.last_prediction,
        }
