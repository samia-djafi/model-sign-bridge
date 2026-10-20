"""
Independent evaluation script for trained Algerian Sign Language model artifacts.

Loads the saved model and evaluate performance on:
1. In-domain Stratified Test Split
2. Avatar-held-out Test Split
3. Camera-held-out Test Split
4. 10-Word MVP Healthcare Vocabulary Sub-evaluation
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.label_parser import LabelParser
from src.evaluation.metrics import compute_classification_metrics
from src.models.classifier import SignClassifier


def evaluate_split(clf: SignClassifier, X_test: np.ndarray, y_test: np.ndarray, target_names: list, split_name: str) -> dict:
    t0 = time.time()
    latencies = []
    y_preds = []
    for i in range(len(X_test)):
        sample = X_test[i : i + 1]
        t_start = time.time()
        pred = clf.predict(sample)[0]
        latencies.append((time.time() - t_start) * 1000)
        y_preds.append(pred)

    y_preds = np.array(y_preds)
    metrics = compute_classification_metrics(y_test, y_preds, target_names, latencies_ms=latencies)

    print(f"\n[{split_name}]")
    print(f"  Test Samples     : {len(X_test)}")
    print(f"  Accuracy         : {metrics['accuracy']*100:.1f}%")
    print(f"  Macro F1         : {metrics['macro_f1']*100:.1f}%")
    print(f"  Macro Precision  : {metrics['macro_precision']*100:.1f}%")
    print(f"  Macro Recall     : {metrics['macro_recall']*100:.1f}%")
    print(f"  Mean Latency     : {metrics['mean_latency_ms']:.2f} ms/sample")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved sign language model.")
    parser.add_argument("--model-dir", default=r"artifacts\classifier", help="Path to model directory")
    parser.add_argument("--features-npz", default=r"data\processed\features_mvp_landmark.npz", help="Path to feature NPZ")
    parser.add_argument("--stratified-manifest", default=r"data\manifests\split_mvp_stratified.csv")
    parser.add_argument("--avatar-manifest", default=r"data\manifests\split_mvp_avatar_heldout.csv")
    parser.add_argument("--camera-manifest", default=r"data\manifests\split_mvp_camera_heldout.csv")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "best_model.joblib")
    config_path = os.path.join(args.model_dir, "config.json")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"[ERROR] Model artifacts not found in {args.model_dir}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    target_names = config["target_names"]
    clf = SignClassifier.load(model_path)
    print("=" * 65)
    print("      INDEPENDENT MODEL EVALUATION — CONNECTSIGN LSA MODEL       ")
    print("=" * 65)
    print(f"Model Name           : {config.get('model_name')}")
    print(f"Feature Dimension    : {config.get('feature_dim')}")
    print(f"Total Vocabulary     : {config.get('num_classes')} classes")
    print(f"Weights Artifact     : {model_path}")

    # Load test data
    npz = np.load(args.features_npz, allow_pickle=True)
    X_all = npz["X"]
    y_all = npz["y"]
    rel_paths = list(npz["rel_paths"])
    path_to_idx = {p: i for i, p in enumerate(rel_paths)}

    def load_test_split(manifest_path):
        df = pd.read_csv(manifest_path, encoding="utf-8-sig")
        test_df = df[df["split"] == "test"]
        indices = [path_to_idx[p] for p in test_df["rel_path"] if p in path_to_idx]
        return X_all[indices], y_all[indices]

    # 1. Stratified In-domain Test
    X_test_strat, y_test_strat = load_test_split(args.stratified_manifest)
    m_strat = evaluate_split(clf, X_test_strat, y_test_strat, target_names, "In-Domain Stratified Test Split (15%)")

    # 2. Avatar Held-out Test
    X_test_av, y_test_av = load_test_split(args.avatar_manifest)
    m_av = evaluate_split(clf, X_test_av, y_test_av, target_names, "Avatar Generalization Test Split (Signer 3)")

    # 3. Camera Held-out Test
    X_test_cam, y_test_cam = load_test_split(args.camera_manifest)
    m_cam = evaluate_split(clf, X_test_cam, y_test_cam, target_names, "Camera Generalization Test Split (Cam 4)")

    # 4. 10-Word MVP Healthcare Subset Evaluation
    words_10 = ["يساعد", "ألم", "خطر", "ماء", "شكرا", "حرارة", "مريض", "رأس", "من فضلك", "نعم"]
    idx_10 = [target_names.index(w) for w in words_10 if w in target_names]
    mask_10 = np.isin(y_test_strat, idx_10)
    if np.any(mask_10):
        X_sub = X_test_strat[mask_10]
        y_sub = y_test_strat[mask_10]
        sub_preds = clf.predict(X_sub)
        acc_10 = float(np.mean(sub_preds == y_sub))
        print(f"\n[10-Word MVP Healthcare Vocabulary Subset (In-domain)]")
        print(f"  Test Samples     : {len(X_sub)}")
        print(f"  Top-1 Accuracy   : {acc_10*100:.1f}%")

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    main()
