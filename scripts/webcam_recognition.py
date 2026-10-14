"""
Standalone webcam experiment for Algerian Sign Language recognition.

Usage:
    python scripts/webcam_recognition.py [--camera-id 0] [--model-dir artifacts/classifier]
    python scripts/webcam_recognition.py --video-simulation data/VDzSL/signer0/cam0/ألم.mp4
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.inference.predictor import VideoSignPredictor
from src.webcam.live_recognizer import LiveSignRecognizer


def run_webcam(
    camera_id: int = 0,
    video_simulation: Optional[str] = None,
    model_dir: str = r"artifacts\classifier",
    threshold: float = 0.35,
    headless: bool = False,
    max_frames: Optional[int] = None,
):
    print("=" * 65)
    print("       ALGERIAN SIGN LANGUAGE (LSA) — LIVE RECOGNIZER       ")
    print("=" * 65)
    print(f"Loading model from {model_dir}...")

    try:
        predictor = VideoSignPredictor(model_dir=model_dir, uncertainty_threshold=threshold)
        recognizer = LiveSignRecognizer(predictor=predictor)
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    is_simulation = video_simulation is not None
    if is_simulation:
        if not os.path.exists(video_simulation):
            print(f"Simulation file does not exist: {video_simulation}")
            return
        cap = cv2.VideoCapture(video_simulation)
        source_name = f"Simulation: {os.path.basename(video_simulation)}"
    else:
        cap = cv2.VideoCapture(camera_id)
        source_name = f"Webcam (Device {camera_id})"

    if not cap.isOpened():
        print(f"\n[ERROR] Failed to open video source: {source_name}")
        print("  - If running without a physical webcam, use --video-simulation to test with a video clip.")
        print("  - Ensure camera privacy permissions are enabled in Windows Settings.")
        return

    print(f"Video stream opened successfully: {source_name}")
    print("Controls: Press 'q' or 'ESC' in the window to safely release camera and exit.\n")

    fps_history = []
    total_frames_processed = 0
    t_prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_simulation:
                    # Cleanly re-open simulation video on EOF
                    cap.release()
                    cap = cv2.VideoCapture(video_simulation)
                    continue
                else:
                    print("Stream terminated or frame dropped.")
                    break

            total_frames_processed += 1
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            fps = 1.0 / dt if dt > 0 else 30.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Process through live sign recognizer
            result = recognizer.process_frame(frame)
            pred = result.get("prediction")

            # Visual overlay
            display = frame.copy()
            h, w = display.shape[:2]

            # Top header bar
            cv2.rectangle(display, (0, 0), (w, 60), (30, 30, 30), -1)
            cv2.putText(
                display,
                f"LSA Healthcare Recognition | FPS: {avg_fps:.1f}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )
            cv2.putText(
                display,
                f"Domain Shift Alert: Avatar Model -> Real Webcam (Research Prototype)",
                (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )

            # Bottom prediction card
            cv2.rectangle(display, (0, h - 110), (w, h), (20, 20, 20), -1)

            if pred is not None:
                lbl = pred["label"]
                conf = pred["confidence"]
                en = pred["meaning_en"]
                fr = pred["meaning_fr"]
                is_unc = pred["is_uncertain"]

                status_color = (0, 0, 255) if is_unc else (0, 255, 0)
                status_text = "UNCERTAIN / RECHECK" if is_unc else "VERIFIED INTENT"

                cv2.putText(
                    display,
                    f"Sign: {lbl} [{status_text}] ({conf:.1%})",
                    (15, h - 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )
                cv2.putText(
                    display,
                    f"Meaning: {en} / {fr} | Latency: {pred['latency_ms']:.1f}ms",
                    (15, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
            else:
                motion_state = "Movement detected (Capturing...)" if result["is_active_motion"] else "Resting / Idle"
                cv2.putText(
                    display,
                    f"Status: {motion_state}",
                    (15, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                )

            # Safety notice banner at bottom
            cv2.putText(
                display,
                "SAFETY: Communication support only. Not for diagnosis or dosage.",
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
            )

            if not headless:
                cv2.imshow("BridgeSign LSA Live Experiment", display)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:  # 'q' or ESC
                    print("Exit signal received. Terminating live experiment...")
                    break
            else:
                if pred is not None:
                    print(f"  Frame {len(fps_history)} | Pred: {pred['label']} ({pred['confidence']:.1%}) | Motion: {result['is_active_motion']}")

            if max_frames and total_frames_processed >= max_frames:
                print(f"Reached max frames ({max_frames}). Stopping experiment.")
                break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print("Camera cleanly released. OpenCV windows closed.")


def main():
    parser = argparse.ArgumentParser(description="Live webcam recognition experiment for LSA.")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video-simulation", default=None, help="Path to video clip for simulated webcam input")
    parser.add_argument("--model-dir", default=r"artifacts\classifier", help="Path to model artifacts")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence uncertainty threshold")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    args = parser.parse_args()

    run_webcam(
        camera_id=args.camera_id,
        video_simulation=args.video_simulation,
        model_dir=args.model_dir,
        threshold=args.threshold,
        headless=args.headless,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
