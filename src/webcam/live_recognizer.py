"""
Live webcam sign language recognizer and clinical stream processor for Algerian Sign Language.

Implements a 7-stage recognition & translation pipeline:
1. Frame capture & buffering.
2. Hand detection & 21-landmark skeleton rendering (via MediaPipe & OpenCV).
3. Temporal windowing & feature extraction.
4. Confidence scoring & uncertainty gating.
5. Token sequence management (stability debounce & refractory transition gating).
6. Clinical sentence assembly (via ClinicalAssembler).
7. Multilingual display (French, Arabic, English) & Pharmacy triage records.
"""

import collections
import time
from typing import Any, Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.features.feature_extractor import LandmarkFeatureExtractor
from src.inference.predictor import VideoSignPredictor
from src.translation.clinical_assembler import ClinicalAssembler, AssemblyResult
from src.webcam.visualizer import HandVisualizer


class LiveSignRecognizer:
    """
    Live stream processor with real-time hand skeleton visualization,
    motion activity gating, temporal debouncing, and clinical sentence assembly.
    """

    def __init__(
        self,
        predictor: VideoSignPredictor,
        window_size: int = 16,
        sample_stride: int = 2,
        motion_threshold: float = 0.0005,
        smoothing_window: int = 3,
        debounce_cooldown_sec: float = 1.5,
        stability_count_threshold: int = 2,
        assembler: Optional[ClinicalAssembler] = None,
    ):
        self.predictor = predictor
        self.window_size = window_size
        self.sample_stride = sample_stride
        self.motion_threshold = motion_threshold
        self.smoothing_window = smoothing_window
        self.debounce_cooldown_sec = debounce_cooldown_sec
        self.stability_count_threshold = stability_count_threshold

        self.visualizer = HandVisualizer(show_labels=True)
        self.assembler = assembler if assembler is not None else ClinicalAssembler(debounce_window_sec=debounce_cooldown_sec)

        # Buffers
        self.frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=window_size)
        self.prediction_history: Deque[Tuple[str, float]] = collections.deque(maxlen=smoothing_window)

        # State tracking
        self.prev_gray: Optional[np.ndarray] = None
        self.last_prediction: Optional[Dict] = None
        self.is_active_motion: bool = False
        self.last_inference_time_ms: float = 0.0
        self.last_hand_latency_ms: float = 0.0

        # Session state
        self.is_paused: bool = False
        self.committed_tokens: List[Dict[str, Any]] = []
        self.last_committed_sign: Optional[str] = None
        self.last_committed_time: float = 0.0

        # Stability counter
        self.candidate_sign: Optional[str] = None
        self.candidate_consecutive_count: int = 0

    def pause(self) -> None:
        """Pause sign commitment."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume sign commitment."""
        self.is_paused = False

    def toggle_pause(self) -> bool:
        """Toggle paused state. Returns new paused state."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def clear_session(self) -> None:
        """Reset the assembled sentence and token history."""
        self.assembler.clear()
        self.committed_tokens.clear()
        self.last_committed_sign = None
        self.candidate_sign = None
        self.candidate_consecutive_count = 0
        self.last_prediction = None

    def finalize_session(self) -> AssemblyResult:
        """Assemble and return the final clinical record and sentences."""
        return self.assembler.assemble()

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single incoming video frame through the 7-stage pipeline:
        - Stage 1: Frame capture and motion computation
        - Stage 2: Hand detection and OpenCV skeleton visualization
        - Stage 3: Temporal buffer sampling and landmark inference
        - Stage 4: Confidence scoring and uncertainty gating
        - Stage 5: Token debouncing and stability verification
        - Stage 6: Clinical slot-filling and sentence assembly
        - Stage 7: Status packaging for HUD display
        """
        now = time.time()
        h, w = frame.shape[:2]

        # Stage 1: Motion activity check
        small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64))
        motion_score = 0.0
        if self.prev_gray is not None:
            diff = np.abs(small_gray.astype(np.float32) - self.prev_gray.astype(np.float32)) / 255.0
            motion_score = float(np.mean(diff))

        self.prev_gray = small_gray
        self.is_active_motion = motion_score > self.motion_threshold
        self.frame_buffer.append(frame.copy())

        # Stage 2: Hand detection & landmark skeleton visualization
        annotated_frame = frame.copy()
        hand_result = None
        num_hands = 0

        if hasattr(self.predictor.extractor, "detect_hands_on_frame"):
            t_hand_start = time.time()
            try:
                hand_result = self.predictor.extractor.detect_hands_on_frame(frame)
                self.last_hand_latency_ms = (time.time() - t_hand_start) * 1000
                if hand_result and hand_result.hand_landmarks:
                    num_hands = len(hand_result.hand_landmarks)
                    annotated_frame = self.visualizer.draw_landmarks_on_frame(annotated_frame, hand_result)
            except Exception as e:
                self.last_hand_latency_ms = 0.0
                hand_result = None

        newly_committed = False

        # Stage 3 & 4: Inference when buffer is full and hands or motion are present
        if len(self.frame_buffer) == self.window_size and (self.is_active_motion or num_hands > 0):
            # Sample uniformly from buffer
            indices = np.linspace(0, self.window_size - 1, num=16, dtype=int)
            sampled_frames = [self.frame_buffer[i] for i in indices]

            t0 = time.time()
            pred_result = self.predictor.predict_frames(sampled_frames)
            self.last_inference_time_ms = (time.time() - t0) * 1000

            if pred_result.get("status") == "SUCCESS":
                top_label = pred_result["predicted_label"]
                top_prob = pred_result["confidence_score"]
                is_uncertain = pred_result["is_uncertain"]

                self.prediction_history.append((top_label, top_prob))

                # Temporal smoothing across recent windows
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
                    "is_uncertain": is_uncertain,
                    "latency_ms": self.last_inference_time_ms,
                    "top_candidates": pred_result.get("top_candidates", []),
                }

                # Stage 5 & 6: Token sequence management and assembly (only if confident and not paused)
                if not is_uncertain and not self.is_paused:
                    if smoothed_label == self.candidate_sign:
                        self.candidate_consecutive_count += 1
                    else:
                        self.candidate_sign = smoothed_label
                        self.candidate_consecutive_count = 1

                    # Stability debounce check
                    if self.candidate_consecutive_count >= self.stability_count_threshold:
                        time_since_last = now - self.last_committed_time
                        is_different_sign = (smoothed_label != self.last_committed_sign)
                        cooldown_expired = (time_since_last >= self.debounce_cooldown_sec)

                        if is_different_sign or cooldown_expired:
                            added = self.assembler.add_token(
                                smoothed_label,
                                confidence=top_prob,
                                timestamp=now,
                            )
                            if added:
                                self.last_committed_sign = smoothed_label
                                self.last_committed_time = now
                                self.committed_tokens.append({
                                    "label": smoothed_label,
                                    "confidence": top_prob,
                                    "timestamp": now,
                                    "meaning_fr": vocab_info.get("meaning_fr", "N/A"),
                                    "meaning_en": vocab_info.get("meaning_en", "N/A"),
                                })
                                newly_committed = True

        # Current live sentence assembly preview
        current_assembly = self.assembler.assemble()

        return {
            "annotated_frame": annotated_frame,
            "hand_result": hand_result,
            "hands_detected": num_hands,
            "motion_score": motion_score,
            "is_active_motion": self.is_active_motion,
            "prediction": self.last_prediction,
            "assembly": current_assembly,
            "committed_tokens": list(self.committed_tokens),
            "newly_committed": newly_committed,
            "is_paused": self.is_paused,
            "hand_latency_ms": self.last_hand_latency_ms,
            "inference_latency_ms": self.last_inference_time_ms,
        }
