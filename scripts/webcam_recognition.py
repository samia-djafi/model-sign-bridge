"""
Standalone webcam experiment for Algerian Sign Language recognition and clinical assembly.

Features:
- Real-time 21 hand landmarks and anatomical skeleton visualization directly over user hands.
- Continuous 7-stage recognition pipeline connecting to ClinicalAssembler.
- Interactive session controls:
    [P] Pause / Resume recognition
    [C] Clear dialogue session
    [F] Finalize and print structured Clinical Pharmacy Record (Fiche Comptoir)
    [Q] / [ESC] Safe exit and camera release

Usage:
    python scripts/webcam_recognition.py [--camera-id 0] [--model-dir artifacts/classifier]
    python scripts/webcam_recognition.py --video-simulation data/VDzSL/signer0/cam0/ألم.mp4
"""

import argparse
import os
import sys
import time
from typing import Optional
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.inference.predictor import VideoSignPredictor
from src.webcam.live_recognizer import LiveSignRecognizer


def print_final_clinical_record(assembly):
    """Print the structured Clinical Pharmacy Record to terminal."""
    print("\n" + "=" * 70)
    print("      FINAL CLINICAL PHARMACY RECORD (FICHE COMPTOIR PHARMACIE)     ")
    print("=" * 70)
    print(f"Urgency Triage Level : {assembly.urgency_level}")
    print(f"Raw Tokens           : {' -> '.join(['[' + t + ']' for t in assembly.raw_tokens]) or 'None'}")
    print(f"French Translation   : \"{assembly.french_sentence}\"")
    print(f"Arabic Translation   : \"{assembly.arabic_sentence}\"")
    print(f"English Translation  : \"{assembly.english_sentence}\"")
    print("-" * 70)
    summary = assembly.clinical_summary
    print(f"Patient State        : {summary.get('patient_state', 'N/A')}")
    print(f"Symptoms Reported    : {', '.join(summary.get('symptomes', [])) or 'None'}")
    print(f"Anatomical Location  : {', '.join(summary.get('siege_anatomique', [])) or 'Not specified'}")
    print(f"Temporality          : {', '.join(summary.get('temporalite', [])) or 'Not specified'}")
    poso = summary.get("posologie", {})
    if poso.get("quantite"):
        print(f"Posology             : {poso.get('quantite')} unit(s) - Frequency: {poso.get('frequence') or 'Single dose'}")
    print("=" * 70 + "\n")


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
    print("Interactive Controls:")
    print("  [P]   Pause / Resume session")
    print("  [C]   Clear tokens and reset sentence")
    print("  [F]   Finalize and print Clinical Pharmacy Record")
    print("  [Q]   Exit cleanly\n")

    fps_history = []
    total_frames_processed = 0
    t_prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_simulation:
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

            # Process through live recognizer (stages 1 through 7)
            res = recognizer.process_frame(frame)
            display = res["annotated_frame"]
            pred = res.get("prediction")
            assembly = res.get("assembly")
            committed = res.get("committed_tokens", [])
            num_hands = res.get("hands_detected", 0)

            h, w = display.shape[:2]

            # 1. Top Header Bar
            cv2.rectangle(display, (0, 0), (w, 55), (25, 25, 25), -1)
            cv2.line(display, (0, 55), (w, 55), (0, 200, 255), 1)

            session_tag = "[PAUSED]" if res["is_paused"] else ("[ACTIVE]" if res["is_active_motion"] else "[RESTING]")
            session_color = (0, 165, 255) if res["is_paused"] else ((0, 255, 0) if res["is_active_motion"] else (180, 180, 180))

            cv2.putText(
                display,
                f"BridgeSign LSA | FPS: {avg_fps:.1f} | Hands: {num_hands} | {session_tag}",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                session_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Prototype Pipeline: Isolated Classifier -> Clinical Slot-Filling Assembler",
                (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

            # 2. Token Stream Ticker (under header)
            token_str = " -> ".join([t["label"] for t in committed[-5:]]) if committed else "None"
            cv2.rectangle(display, (0, 56), (w, 82), (15, 15, 15), -1)
            cv2.putText(
                display,
                f"Committed Tokens: {token_str}",
                (12, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 220),
                1,
                cv2.LINE_AA,
            )

            # 3. Bottom Clinical Translation Card
            card_h = 135
            cv2.rectangle(display, (0, h - card_h), (w, h), (18, 18, 18), -1)
            cv2.line(display, (0, h - card_h), (w, h - card_h), (0, 200, 255), 1)

            # Current prediction line
            if pred is not None:
                lbl = pred["label"]
                conf = pred["confidence"]
                en = pred["meaning_en"]
                fr = pred["meaning_fr"]
                is_unc = pred["is_uncertain"]

                status_color = (0, 0, 255) if is_unc else (0, 255, 0)
                status_text = "UNCERTAIN / RECHECK" if is_unc else "VERIFIED"

                cv2.putText(
                    display,
                    f"Current Sign: {lbl} [{status_text}] ({conf:.1%}) -> {fr} / {en}",
                    (12, h - card_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "Current Sign: [Monitoring motion & hand tracking...]",
                    (12, h - card_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )

            # Clinical sentence translation lines
            if assembly is not None and assembly.raw_tokens:
                urg_color = (0, 0, 255) if assembly.urgency_level == "CRITICAL" else ((0, 165, 255) if assembly.urgency_level == "ATTENTION" else (0, 255, 0))
                cv2.putText(
                    display,
                    f"French  : {assembly.french_sentence[:75]}",
                    (12, h - card_h + 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Triage  : [{assembly.urgency_level}] | Tokens: {len(assembly.raw_tokens)}",
                    (12, h - card_h + 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    urg_color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "French  : (No signs committed yet. Perform signs in camera view)",
                    (12, h - card_h + 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (140, 140, 140),
                    1,
                    cv2.LINE_AA,
                )

            # Controls help bar at very bottom
            cv2.rectangle(display, (0, h - 26), (w, h), (10, 10, 10), -1)
            cv2.putText(
                display,
                "[P] Pause/Resume | [C] Clear | [F] Finalize Record | [Q] Exit",
                (12, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            # Newly committed notification in console
            if res.get("newly_committed"):
                last_tok = committed[-1]
                print(f"  + Committed sign: '{last_tok['label']}' ({last_tok['meaning_fr']}) | Conf: {last_tok['confidence']:.1%}")

            if not headless:
                cv2.imshow("BridgeSign LSA Live Pipeline", display)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), 27]:
                    print("Exit requested. Closing...")
                    break
                elif key in [ord("p"), ord("P")]:
                    paused = recognizer.toggle_pause()
                    print(f"  [Action] Session {'PAUSED' if paused else 'RESUMED'}")
                elif key in [ord("c"), ord("C")]:
                    recognizer.clear_session()
                    print("  [Action] Session CLEARED (buffer reset)")
                elif key in [ord("f"), ord("F")]:
                    final_rec = recognizer.finalize_session()
                    print_final_clinical_record(final_rec)
            else:
                if pred is not None and not pred["is_uncertain"]:
                    print(f"  Frame {total_frames_processed} | Sign: {pred['label']} ({pred['confidence']:.1%}) | Hands: {num_hands}")

            if max_frames and total_frames_processed >= max_frames:
                print(f"Reached max frames ({max_frames}). Stopping experiment.")
                break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        final_summary = recognizer.finalize_session()
        if final_summary.raw_tokens:
            print_final_clinical_record(final_summary)
        print("Camera cleanly released. OpenCV windows closed.")


def main():
    parser = argparse.ArgumentParser(description="Live webcam recognition & clinical assembly experiment.")
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
