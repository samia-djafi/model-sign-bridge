"""
Unit tests for BridgeSign HTTP Recognition API Server and Phrasebook Matcher.
"""

import json
import threading
import time
import urllib.request
import pytest

from src.server.api import BridgeSignRequestHandler
from http.server import ThreadingHTTPServer
from src.inference.predictor import VideoSignPredictor
from src.translation.phrasebook_matcher import PhrasebookMatcher, SEED_PHRASES


@pytest.fixture(scope="module")
def test_server():
    """Spin up local test server on port 8999."""
    port = 8999
    predictor = VideoSignPredictor(model_dir=r"artifacts\classifier")
    matcher = PhrasebookMatcher(predictor.target_names)

    BridgeSignRequestHandler.predictor = predictor
    BridgeSignRequestHandler.matcher = matcher

    server = ThreadingHTTPServer(("127.0.0.1", port), BridgeSignRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)

    base_url = f"http://127.0.0.1:{port}"
    yield base_url

    server.shutdown()
    server.server_close()


def test_phrasebook_matcher_coverage():
    """Verify phrasebook matcher covers all 45 phrases and has expected categories."""
    matcher = PhrasebookMatcher()
    report = matcher.generate_coverage_report()
    assert report["total_phrases"] == 45
    summary = report["support_summary"]
    assert summary["FULL_RECOGNITION"] + summary["PARTIAL_RECOGNITION"] + summary["PHRASEBOOK_FALLBACK"] == 45


def test_phrasebook_matching_exact():
    """Verify exact match for single-sign phrases."""
    matcher = PhrasebookMatcher()
    matches = matcher.match_signs(["HEADACHE"])
    assert len(matches) > 0
    assert matches[0]["phrase"]["id"] == "ph_headache"
    assert matches[0]["match_type"] == "EXACT"

    # Match using Arabic label
    matches_ar = matcher.match_signs(["ألم"])
    assert len(matches_ar) > 0
    assert matches_ar[0]["phrase"]["id"] == "ph_pain"


def test_api_health(test_server):
    """Verify /health endpoint returns 200 and metadata."""
    req = urllib.request.Request(f"{test_server}/health")
    with urllib.request.urlopen(req, timeout=3) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert data["phrasebook_phrases_count"] == 45


def test_api_phrases(test_server):
    """Verify /v1/phrases endpoint returns all 45 phrases."""
    req = urllib.request.Request(f"{test_server}/v1/phrases")
    with urllib.request.urlopen(req, timeout=3) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["total_phrases"] == 45
        assert len(data["phrase_evaluations"]) == 45


def test_api_recognize_empty(test_server):
    """Verify /v1/recognize handles empty payload gracefully."""
    payload = json.dumps({"lang": "fr", "frames": []}).encode("utf-8")
    req = urllib.request.Request(
        f"{test_server}/v1/recognize",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["signs"] == []
        assert data["sequence"] == []
        assert data["adapter"] == "live"


def test_api_recognize_landmarks(test_server):
    """Verify /v1/recognize processes 16 synthetic LandmarkFrames and returns standard response."""
    frames = [
        {
            "t": i * 60,
            "hands": [[{"x": 0.5 + 0.005 * j, "y": 0.5 + 0.005 * j, "z": 0.0} for j in range(21)]],
            "pose": [{"x": 0.5, "y": 0.3, "z": 0.0} for _ in range(33)],
        }
        for i in range(16)
    ]
    payload = json.dumps({"lang": "fr", "frames": frames}).encode("utf-8")
    req = urllib.request.Request(
        f"{test_server}/v1/recognize",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert "overallConfidence" in data
        assert "level" in data
        assert data["frames"] == 16
        assert "model" in data
        assert data["adapter"] == "live"


def test_api_recognize_base64_images(test_server):
    """Verify /v1/recognize processes base64 image frames."""
    import base64
    import cv2
    import numpy as np

    # Generate small 16 dummy frames
    frames = []
    for i in range(16):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
        frames.append(b64)

    payload = json.dumps({"lang": "fr", "frames": frames}).encode("utf-8")
    req = urllib.request.Request(
        f"{test_server}/v1/recognize",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        assert data["frames"] == 16
        assert data["adapter"] == "live"
        # Since blank frames have no hands, confidence should be low and rejected
        assert data["isUncertain"] is True
        assert data["rejectionMessage"] == "Sign not recognized. Please try again."


def test_api_invalid_json(test_server):
    """Verify /v1/recognize responds with 400 on invalid JSON."""
    import urllib.error
    req = urllib.request.Request(
        f"{test_server}/v1/recognize",
        data=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=3)
    assert exc_info.value.code == 400


def test_api_404_not_found(test_server):
    """Verify unknown endpoint returns 404."""
    import urllib.error
    req = urllib.request.Request(f"{test_server}/v1/non_existent_route")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=3)
    assert exc_info.value.code == 404

