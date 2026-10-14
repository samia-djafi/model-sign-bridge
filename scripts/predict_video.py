"""
Command-line tool to predict Algerian Sign Language signs from a video file.

Usage:
    python scripts/predict_video.py --video path/to/sample.mp4 [--threshold 0.35] [--json]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.inference.predictor import VideoSignPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict Algerian Sign Language sign from a video.")
    parser.add_argument("--video", required=True, help="Path to input MP4 video file")
    parser.add_argument("--model-dir", default=r"artifacts\classifier", help="Path to classifier artifacts directory")
    parser.add_argument("--threshold", type=float, default=0.35, help="Uncertainty rejection threshold (default: 0.35)")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    args = parser.parse_args()

    predictor = VideoSignPredictor(
        model_dir=args.model_dir,
        uncertainty_threshold=args.threshold,
    )

    result = predictor.predict_video(args.video)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 60)
    print("      ALGERIAN SIGN LANGUAGE RECOGNITION (LSA)      ")
    print("=" * 60)
    print(f"Video File           : {result.get('video_path', args.video)}")
    print(f"Model Version        : {result.get('model_version', 'N/A')}")
    print(f"Decision Status      : {result.get('decision', 'N/A')}")
    print("-" * 60)

    if result.get("status") == "ERROR":
        print(f"ERROR: {result.get('error')}")
        return

    top1 = result["predicted_label"]
    score = result["confidence_score"]
    meaning_en = result["verified_meaning_en"]
    meaning_fr = result["verified_meaning_fr"]
    category = result["category"]

    print(f"Predicted Sign       : {top1} (Arabic)")
    print(f"Confidence Score     : {score:.1%} ({result.get('score_interpretation')})")
    print(f"Verified Meaning (EN): {meaning_en}")
    print(f"Verified Meaning (FR): {meaning_fr}")
    print(f"Sign Category        : {category}")
    print(f"Pharmacy Relevance   : {result.get('pharmacy_relevance')}")

    print("\nTop Alternative Candidates:")
    for rank, cand in enumerate(result["top_candidates"], 1):
        print(f"  {rank}. {cand['label']} ({cand['score']:.1%}) -> {cand['meaning_en']} / {cand['meaning_fr']}")

    if result["warnings"]:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  [!] {w}")

    print("-" * 60)
    print(f"[!] SAFETY NOTICE: {result['medical_safety_notice']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
