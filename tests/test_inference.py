"""
Unit tests for VideoSignPredictor inference engine.
"""

import os
import tempfile
import pytest
from src.inference.predictor import VideoSignPredictor


@pytest.fixture
def predictor():
    model_dir = r"artifacts\classifier"
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        pytest.skip(f"Model config not found at {config_path}. Train model first.")
    return VideoSignPredictor(model_dir=model_dir, uncertainty_threshold=0.35)


def test_predict_missing_file(predictor):
    res = predictor.predict_video("non_existent_sample.mp4")
    assert res["status"] == "ERROR"
    assert res["is_uncertain"] is True
    assert "File does not exist" in res["error"]


def test_predict_corrupted_file(predictor):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"NOT_A_VALID_MP4_HEADER_CORRUPTED_DATA")
        tmp_path = f.name

    try:
        res = predictor.predict_video(tmp_path)
        assert res["status"] == "ERROR"
        assert res["is_uncertain"] is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_predict_real_sample(predictor):
    sample_path = r"data\VDzSL\signer0\cam0\1.mp4"
    if not os.path.exists(sample_path):
        pytest.skip(f"Test sample not found: {sample_path}")

    res = predictor.predict_video(sample_path)
    assert res["status"] == "SUCCESS"
    assert 0.0 <= res["confidence_score"] <= 1.0
    assert len(res["top_candidates"]) == 3
    assert "medical_safety_notice" in res
    assert res["decision"] in ["ACCEPT", "REJECT / UNCERTAIN"]
