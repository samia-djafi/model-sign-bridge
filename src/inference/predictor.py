"""
Video-level sign language prediction module for Algerian Sign Language models.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.feature_extractor import PixelMotionFeatureExtractor
from src.models.classifier import SignClassifier


class VideoSignPredictor:
    """
    Inference engine that loads model weights, applies consistent preprocessing,
    and returns predictions with uncertainty calibration and verified meanings.
    """

    def __init__(
        self,
        model_dir: str = r"artifacts\classifier",
        vocab_csv: str = r"data\manifests\medical_vocabulary.csv",
        uncertainty_threshold: float = 0.35,
    ):
        self.model_dir = model_dir
        self.uncertainty_threshold = uncertainty_threshold

        config_path = os.path.join(model_dir, "config.json")
        model_path = os.path.join(model_dir, "best_model.joblib")

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.classifier = SignClassifier.load(model_path)
        self.target_names = self.config["target_names"]

        # Initialize feature extractor with exact parameters recorded in config
        prep_cfg = self.config.get("preprocessing_config", {})
        self.extractor = PixelMotionFeatureExtractor(
            num_frames=prep_cfg.get("num_frames", 16),
            target_size=tuple(prep_cfg.get("target_size", [32, 32])),
        )

        # Load verified vocabulary lookup
        self.vocab_lookup = {}
        if os.path.exists(vocab_csv):
            df_vocab = pd.read_csv(vocab_csv, encoding="utf-8-sig")
            for _, r in df_vocab.iterrows():
                self.vocab_lookup[r["original_label"]] = {
                    "meaning_en": r["meaning_en"],
                    "meaning_fr": r["meaning_fr"],
                    "category": r["category"],
                    "relevance": r["pharmacy_relevance"],
                    "rationale": r["relevance_rationale"],
                }

    def predict_video(self, video_path: str) -> Dict:
        """
        Run inference on a single video file.

        Returns:
            Dictionary with prediction results, probabilities, uncertainty flag,
            and clinical safety warnings.
        """
        if not os.path.exists(video_path):
            return {
                "status": "ERROR",
                "error": f"File does not exist: {video_path}",
                "is_uncertain": True,
            }

        warnings = []
        try:
            features = self.extractor.extract_from_path(video_path)
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Video decoding or preprocessing failed: {str(e)}",
                "is_uncertain": True,
            }

        # Predict probabilities
        X = np.expand_dims(features, axis=0)
        probs = self.classifier.predict_proba(X)[0]

        top_indices = np.argsort(probs)[::-1]
        top1_idx = top_indices[0]
        top1_label = self.target_names[top1_idx]
        top1_prob = float(probs[top1_idx])

        # Top-3 candidates
        top_candidates = []
        for i in top_indices[:3]:
            lbl = self.target_names[i]
            top_candidates.append({
                "label": lbl,
                "score": float(probs[i]),
                "meaning_en": self.vocab_lookup.get(lbl, {}).get("meaning_en", "N/A"),
                "meaning_fr": self.vocab_lookup.get(lbl, {}).get("meaning_fr", "N/A"),
            })

        # Uncertainty evaluation
        is_uncertain = top1_prob < self.uncertainty_threshold
        if is_uncertain:
            warnings.append(
                f"Confidence score ({top1_prob:.1%}) is below certainty threshold ({self.uncertainty_threshold:.1%})."
            )

        vocab_info = self.vocab_lookup.get(top1_label, {})

        result = {
            "status": "SUCCESS",
            "video_path": video_path,
            "predicted_label": top1_label,
            "confidence_score": top1_prob,
            "score_interpretation": "Normalized softmax probability from calibrated linear/neural classifier",
            "is_uncertain": is_uncertain,
            "decision": "REJECT / UNCERTAIN" if is_uncertain else "ACCEPT",
            "verified_meaning_en": vocab_info.get("meaning_en", "N/A"),
            "verified_meaning_fr": vocab_info.get("meaning_fr", "N/A"),
            "category": vocab_info.get("category", "General"),
            "pharmacy_relevance": vocab_info.get("relevance", "N/A"),
            "top_candidates": top_candidates,
            "model_version": self.config.get("model_name", "Unknown"),
            "warnings": warnings,
            "medical_safety_notice": "COMMUNICATION SUPPORT ONLY. Do not use for medical diagnosis, dosage prescription, or autonomous healthcare decisions.",
        }
        return result
