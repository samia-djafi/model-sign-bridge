"""
Unit tests for landmark visualization, landmark extraction, and live session recognition.
"""

import os
import numpy as np
import pytest
import cv2

from src.features.feature_extractor import LandmarkFeatureExtractor
from src.inference.predictor import VideoSignPredictor
from src.webcam.visualizer import HandVisualizer
from src.webcam.live_recognizer import LiveSignRecognizer
from src.translation.clinical_assembler import ClinicalAssembler


def test_hand_visualizer_none_or_empty():
    visualizer = HandVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Should safely return untouched frame on None or empty result
    out1 = visualizer.draw_landmarks_on_frame(frame.copy(), None)
    assert out1.shape == (480, 640, 3)

    class DummyEmptyResult:
        hand_landmarks = []
        handedness = []

    out2 = visualizer.draw_landmarks_on_frame(frame.copy(), DummyEmptyResult())
    assert out2.shape == (480, 640, 3)


def test_hand_visualizer_mock_landmarks():
    visualizer = HandVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class MockCategory:
        category_name = "Right"
        score = 0.95

    class MockResult:
        hand_landmarks = [[MockLandmark(i / 21.0, (20 - i) / 21.0) for i in range(21)]]
        handedness = [[MockCategory()]]

    out = visualizer.draw_landmarks_on_frame(frame, MockResult())
    assert out.shape == (480, 640, 3)
    # Ensure drawing modified some pixels
    assert np.any(out > 0)


def test_landmark_feature_extractor_synthetic():
    ext = LandmarkFeatureExtractor(num_frames=8)
    dummy_frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(8)]
    feat = ext.extract_from_frames(dummy_frames)

    assert isinstance(feat, np.ndarray)
    assert feat.dtype == np.float32
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()
    ext.close()


def test_live_recognizer_session_management():
    model_dir = r"artifacts\classifier"
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        pytest.skip("Model not found in artifacts/classifier")

    predictor = VideoSignPredictor(model_dir=model_dir, uncertainty_threshold=0.35)
    recognizer = LiveSignRecognizer(predictor=predictor, debounce_cooldown_sec=1.0)

    # 1. Initial state
    assert not recognizer.is_paused
    assert len(recognizer.committed_tokens) == 0

    # 2. Pause & Resume
    recognizer.pause()
    assert recognizer.is_paused
    recognizer.resume()
    assert not recognizer.is_paused
    assert recognizer.toggle_pause() is True
    assert recognizer.is_paused

    # 3. Clear session
    recognizer.assembler.add_token("ألم")
    recognizer.committed_tokens.append({"label": "ألم"})
    assert len(recognizer.committed_tokens) == 1

    recognizer.clear_session()
    assert len(recognizer.committed_tokens) == 0
    assert len(recognizer.assembler.tokens) == 0

    # 4. Finalize session
    recognizer.assembler.add_token("رأس")
    final_res = recognizer.finalize_session()
    assert "رأس" in final_res.raw_tokens
