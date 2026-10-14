"""
Train and benchmark isolated sign classifiers on Algerian Sign Language MVP dataset.

Compares:
- Linear SVM
- Logistic Regression
- Random Forest
- Multi-Layer Perceptron (MLP)

Evaluates on:
1. In-domain Stratified Split
2. Avatar-Held-Out Generalization (Test: signer3)
3. Camera-Held-Out Generalization (Test: cam4)

Saves best model artifacts, training logs, and evaluation reports.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.data.label_parser import LabelParser
from src.evaluation.metrics import compute_classification_metrics
from src.models.classifier import SignClassifier


def benchmark_model(
    model_name: str,
    clf: SignClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list,
) -> dict:
    print(f"\n--- Training {model_name} ---")
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_dur = time.time() - t0

    # Validation evaluation
    y_val_pred = clf.predict(X_val)
    val_metrics = compute_classification_metrics(y_val, y_val_pred, target_names)

    # Test evaluation & latency benchmark
    latencies = []
    y_test_pred = []
    for i in range(len(X_test)):
        sample = X_test[i : i + 1]
        t_start = time.time()
        pred = clf.predict(sample)[0]
        latencies.append((time.time() - t_start) * 1000)
        y_test_pred.append(pred)

    y_test_pred = np.array(y_test_pred)
    test_metrics = compute_classification_metrics(y_test, y_test_pred, target_names, latencies_ms=latencies)

    print(f"  Training time    : {train_dur:.2f}s")
    print(f"  Val Accuracy     : {val_metrics['accuracy']*100:.1f}% | Macro F1: {val_metrics['macro_f1']*100:.1f}%")
    print(f"  Test Accuracy    : {test_metrics['accuracy']*100:.1f}% | Macro F1: {test_metrics['macro_f1']*100:.1f}%")
    print(f"  Inference Latency: {test_metrics['mean_latency_ms']:.2f} ms/sample")

    return {
        "model_name": model_name,
        "train_duration_sec": train_dur,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "classifier_instance": clf,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate isolated sign classifiers.")
    parser.add_argument(
        "--features-npz",
        default=r"data\processed\features_mvp_pixel.npz",
        help="Path to cached feature NPZ file",
    )
    parser.add_argument(
        "--stratified-manifest",
        default=r"data\manifests\split_mvp_stratified.csv",
        help="Path to stratified split manifest",
    )
    parser.add_argument(
        "--avatar-manifest",
        default=r"data\manifests\split_mvp_avatar_heldout.csv",
        help="Path to avatar held-out manifest",
    )
    parser.add_argument(
        "--camera-manifest",
        default=r"data\manifests\split_mvp_camera_heldout.csv",
        help="Path to camera held-out manifest",
    )
    parser.add_argument(
        "--labels-json",
        default=r"artifacts\label_mappings\mvp_classes.json",
        help="Path to MVP classes JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=r"artifacts\classifier",
        help="Directory to save model artifacts",
    )
    args = parser.parse_args()

    # 1. Load data and labels
    print(f"Loading cached features from {args.features_npz}...")
    npz = np.load(args.features_npz, allow_pickle=True)
    X_all = npz["X"]
    y_all = npz["y"]
    rel_paths = list(npz["rel_paths"])

    label_parser = LabelParser.load(args.labels_json)
    target_names = [label_parser.decode(i) for i in range(label_parser.num_classes)]
    print(f"Total samples: {len(X_all)} | Feature dimension: {X_all.shape[1]} | Classes: {len(target_names)}")

    # Map relative paths to indices in X_all
    path_to_idx = {p: i for i, p in enumerate(rel_paths)}

    def get_split_arrays(manifest_path):
        df = pd.read_csv(manifest_path, encoding="utf-8-sig")
        train_idx = [path_to_idx[p] for p in df[df["split"] == "train"]["rel_path"] if p in path_to_idx]
        val_idx = [path_to_idx[p] for p in df[df["split"] == "val"]["rel_path"] if p in path_to_idx]
        test_idx = [path_to_idx[p] for p in df[df["split"] == "test"]["rel_path"] if p in path_to_idx]
        return (
            X_all[train_idx], y_all[train_idx],
            X_all[val_idx], y_all[val_idx],
            X_all[test_idx], y_all[test_idx],
        )

    # 2. Benchmark on Stratified Split
    print("\n=======================================================")
    print("PROTOCOL 1: IN-DOMAIN STRATIFIED EVALUATION (70/15/15)")
    print("=======================================================")
    X_train, y_train, X_val, y_val, X_test, y_test = get_split_arrays(args.stratified_manifest)
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    candidate_models = {
        "Linear SVM": SignClassifier("linear_svm", C=0.5),
        "Logistic Regression": SignClassifier("logistic_regression", C=1.0, max_iter=1000),
        "Random Forest": SignClassifier("random_forest", n_estimators=150, max_depth=16),
        "MLP (256-128)": SignClassifier("mlp", hidden_layer_sizes=(256, 128), max_iter=250),
    }

    benchmark_results = {}
    best_acc = -1.0
    best_model_name = None
    best_clf = None

    for name, clf in candidate_models.items():
        res = benchmark_model(name, clf, X_train, y_train, X_val, y_val, X_test, y_test, target_names)
        benchmark_results[name] = res
        acc = res["test_metrics"]["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_clf = res["classifier_instance"]

    # 3. Benchmark best model on Avatar-Held-Out Split
    print("\n=======================================================")
    print("PROTOCOL 2: AVATAR GENERALIZATION EVALUATION (Held-out: signer3)")
    print("=======================================================")
    X_tr_av, y_tr_av, X_val_av, y_val_av, X_te_av, y_te_av = get_split_arrays(args.avatar_manifest)
    print(f"Train samples: {len(X_tr_av)} (signer0, signer1) | Val: {len(X_val_av)} (signer2) | Test: {len(X_te_av)} (signer3)")

    clf_avatar = SignClassifier(best_clf.model_type, **best_clf.kwargs)
    res_avatar = benchmark_model(f"{best_model_name} [Avatar Held-out]", clf_avatar, X_tr_av, y_tr_av, X_val_av, y_val_av, X_te_av, y_te_av, target_names)

    # 4. Benchmark best model on Camera-Held-Out Split
    print("\n=======================================================")
    print("PROTOCOL 3: CAMERA GENERALIZATION EVALUATION (Held-out: cam4)")
    print("=======================================================")
    X_tr_cam, y_tr_cam, X_val_cam, y_val_cam, X_te_cam, y_te_cam = get_split_arrays(args.camera_manifest)
    print(f"Train samples: {len(X_tr_cam)} (cam0..cam2) | Val: {len(X_val_cam)} (cam3) | Test: {len(X_te_cam)} (cam4)")

    clf_cam = SignClassifier(best_clf.model_type, **best_clf.kwargs)
    res_cam = benchmark_model(f"{best_model_name} [Camera Held-out]", clf_cam, X_tr_cam, y_tr_cam, X_val_cam, y_val_cam, X_te_cam, y_te_cam, target_names)

    # 5. Save best model artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = best_clf.save(args.output_dir, "best_model.joblib")
    model_size_kb = os.path.getsize(model_path) / 1024

    config = {
        "model_name": best_model_name,
        "model_type": best_clf.model_type,
        "feature_dim": X_all.shape[1],
        "num_classes": len(target_names),
        "target_names": target_names,
        "artifact_file": "best_model.joblib",
        "artifact_size_kb": model_size_kb,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": best_acc,
        "test_macro_f1": benchmark_results[best_model_name]["test_metrics"]["macro_f1"],
        "mean_latency_ms": benchmark_results[best_model_name]["test_metrics"]["mean_latency_ms"],
        "preprocessing_config": {
            "num_frames": 16,
            "target_size": [32, 32],
            "grayscale": True,
            "normalize": True,
            "feature_type": "spatio_temporal_pixel_motion",
        },
    }

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 6. Save comprehensive evaluation report
    report_data = {
        "best_model": best_model_name,
        "comparison_table": {
            name: {
                "train_time_sec": res["train_duration_sec"],
                "val_accuracy": res["val_metrics"]["accuracy"],
                "val_macro_f1": res["val_metrics"]["macro_f1"],
                "test_accuracy": res["test_metrics"]["accuracy"],
                "test_macro_f1": res["test_metrics"]["macro_f1"],
                "test_macro_precision": res["test_metrics"]["macro_precision"],
                "test_macro_recall": res["test_metrics"]["macro_recall"],
                "mean_latency_ms": res["test_metrics"]["mean_latency_ms"],
            }
            for name, res in benchmark_results.items()
        },
        "generalization_evaluation": {
            "in_domain_accuracy": best_acc,
            "in_domain_macro_f1": benchmark_results[best_model_name]["test_metrics"]["macro_f1"],
            "avatar_held_out_accuracy": res_avatar["test_metrics"]["accuracy"],
            "avatar_held_out_macro_f1": res_avatar["test_metrics"]["macro_f1"],
            "camera_held_out_accuracy": res_cam["test_metrics"]["accuracy"],
            "camera_held_out_macro_f1": res_cam["test_metrics"]["macro_f1"],
        },
        "per_class_metrics": benchmark_results[best_model_name]["test_metrics"]["per_class"],
    }

    reports_dir = r"artifacts\reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_json_path = os.path.join(reports_dir, "evaluation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Save confusion matrix to CSV
    cm = benchmark_results[best_model_name]["test_metrics"]["confusion_matrix"]
    df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_csv_path = os.path.join(reports_dir, "confusion_matrix.csv")
    df_cm.to_csv(cm_csv_path, encoding="utf-8-sig")

    print("\n=======================================================")
    print("FINAL BENCHMARK SUMMARY")
    print("=======================================================")
    print(f"Best Model                  : {best_model_name}")
    print(f"Artifact Saved To           : {model_path} ({model_size_kb:.1f} KB)")
    print(f"Config Saved To             : {config_path}")
    print(f"Evaluation Report Saved To  : {report_json_path}")
    print(f"Confusion Matrix Saved To   : {cm_csv_path}")
    print(f"In-Domain Test Accuracy     : {best_acc*100:.2f}%")
    print(f"Avatar Held-out Accuracy    : {res_avatar['test_metrics']['accuracy']*100:.2f}%")
    print(f"Camera Held-out Accuracy    : {res_cam['test_metrics']['accuracy']*100:.2f}%")
    print(f"Average Prediction Latency  : {benchmark_results[best_model_name]['test_metrics']['mean_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
