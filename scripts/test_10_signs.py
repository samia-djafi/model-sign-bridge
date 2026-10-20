"""
Test the 10-word MVP sign recognition vocabulary against genuine dataset videos.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference.predictor import VideoSignPredictor

def test_10():
    predictor = VideoSignPredictor(uncertainty_threshold=0.30)
    
    signs = [
        ("HELP", "يساعد"),
        ("PAIN", "ألم"),
        ("EMERGENCY", "خطر"),
        ("WATER", "ماء"),
        ("THANK_YOU", "شكرا"),
        ("FEVER", "حرارة"),
        ("SICK", "مريض"),
        ("HEAD", "رأس"),
        ("PLEASE", "من فضلك"),
        ("YES", "نعم"),
    ]

    print("=" * 70)
    print(" EVALUATION OF 10-WORD MVP VOCABULARY ON REAL DATASET VIDEOS (VDzSL)")
    print("=" * 70)
    
    total = 0
    correct = 0
    
    for gloss, ar in signs:
        vid_path = os.path.join("data", "VDzSL", "signer0", "cam0", f"{ar}.mp4")
        if not os.path.exists(vid_path):
            print(f"[-] {gloss} ({ar}): Video file not found: {vid_path}")
            continue
            
        res = predictor.predict_video(vid_path)
        pred = res.get("predicted_label")
        conf = res.get("confidence_score", 0.0)
        dec = res.get("decision", "N/A")
        is_match = (pred == ar)
        total += 1
        if is_match:
            correct += 1
            
        print(f"[{'PASS' if is_match else 'FAIL'}] {gloss:<12} (True: {ar:<10}) -> Pred: {pred:<10} | Conf: {conf:.1%} | Dec: {dec}")

    print("-" * 70)
    print(f"Overall Accuracy on signer0/cam0: {correct}/{total} ({correct/total:.1%})")
    print("=" * 70)

if __name__ == "__main__":
    test_10()
