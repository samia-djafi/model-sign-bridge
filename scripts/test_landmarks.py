"""
Empirical test for landmark extraction (MediaPipe Tasks) on synthetic 3D avatars.

Evaluates hand and pose detection reliability across:
- Multiple avatars (signer0, signer1, signer2, signer3)
- Multiple camera angles (cam0, cam1, cam2, cam3, cam4)
- Sample signs (e.g., 'ألم', 'طبيب أسنان', '1', 'ماء', 'يساعد')

Reports:
- Frame detection rate (% of frames with hands detected)
- Extraction latency per frame (ms)
- Failure modes on synthetic textures and camera angles
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


def evaluate_video_landmarks(video_path: str, hand_detector, pose_detector):
    """
    Process a video frame-by-frame and compute landmark detection metrics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = 0
    hands_detected_frames = 0
    both_hands_detected = 0
    pose_detected_frames = 0
    latencies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        t0 = time.time()
        # Hand detection
        hand_result = hand_detector.detect(mp_img)
        # Pose detection
        pose_result = pose_detector.detect(mp_img)
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

        num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        if num_hands > 0:
            hands_detected_frames += 1
        if num_hands >= 2:
            both_hands_detected += 1

        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose_detected_frames += 1

    cap.release()

    if total_frames == 0:
        return None

    return {
        "video": os.path.basename(video_path),
        "total_frames": total_frames,
        "hand_detection_pct": (hands_detected_frames / total_frames) * 100,
        "both_hands_pct": (both_hands_detected / total_frames) * 100,
        "pose_detection_pct": (pose_detected_frames / total_frames) * 100,
        "mean_latency_ms": np.mean(latencies),
    }


def main():
    hand_model_path = r"artifacts/models/hand_landmarker.task"
    pose_model_path = r"artifacts/models/pose_landmarker_lite.task"

    if not os.path.exists(hand_model_path) or not os.path.exists(pose_model_path):
        print("Required model task files not found in artifacts/models/")
        sys.exit(1)

    print("Initializing MediaPipe Hand & Pose detectors (XNNPACK CPU backend)...")
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    hand_detector = vision.HandLandmarker.create_from_options(hand_options)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    base = r"d:\bridge_sign_model\data\VDzSL"
    test_cases = [
        # Frontal view across 4 avatars
        ("signer0", "cam0", "ألم.mp4"),
        ("signer1", "cam0", "ألم.mp4"),
        ("signer2", "cam0", "ألم.mp4"),
        ("signer3", "cam0", "ألم.mp4"),
        # Different camera angles on signer0
        ("signer0", "cam0", "طبيب أسنان.mp4"),
        ("signer0", "cam1", "طبيب أسنان.mp4"),  # -30 deg horizontal
        ("signer0", "cam2", "طبيب أسنان.mp4"),  # +30 deg horizontal
        ("signer0", "cam3", "طبيب أسنان.mp4"),  # +15 deg elevated
        ("signer0", "cam4", "طبيب أسنان.mp4"),  # -15 deg depressed
        # Different signs
        ("signer1", "cam0", "ماء.mp4"),
        ("signer2", "cam0", "يساعد.mp4"),
        ("signer3", "cam0", "1.mp4"),
    ]

    print(f"\nEvaluating landmark detection across {len(test_cases)} representative test clips:\n")
    print(f"{'Avatar/Cam':<18} | {'Sign':<14} | {'Frames':<7} | {'Hand %':<8} | {'2-Hands %':<10} | {'Pose %':<8} | {'Latency':<10}")
    print("-" * 88)

    all_hand_pcts = []
    all_pose_pcts = []
    all_latencies = []

    for signer, cam, sign_file in test_cases:
        video_path = os.path.join(base, signer, cam, sign_file)
        if not os.path.exists(video_path):
            print(f"Skipping missing: {video_path}")
            continue

        res = evaluate_video_landmarks(video_path, hand_detector, pose_detector)
        if res:
            tag = f"{signer}/{cam}"
            sign_name = os.path.splitext(sign_file)[0]
            print(f"{tag:<18} | {sign_name:<14} | {res['total_frames']:<7} | {res['hand_detection_pct']:>6.1f}% | {res['both_hands_pct']:>8.1f}% | {res['pose_detection_pct']:>6.1f}% | {res['mean_latency_ms']:>6.1f} ms")
            all_hand_pcts.append(res["hand_detection_pct"])
            all_pose_pcts.append(res["pose_detection_pct"])
            all_latencies.append(res["mean_latency_ms"])

    hand_detector.close()
    pose_detector.close()

    print("-" * 88)
    if all_hand_pcts:
        print(f"Overall Mean Hand Detection Rate : {np.mean(all_hand_pcts):.1f}%")
        print(f"Overall Mean Pose Detection Rate : {np.mean(all_pose_pcts):.1f}%")
        print(f"Overall Mean Latency (CPU)       : {np.mean(all_latencies):.1f} ms/frame (~{1000/np.mean(all_latencies):.1f} FPS)")


if __name__ == "__main__":
    main()
