"""
Video-level and frame-sequence sign language prediction module for Algerian Sign Language models.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.features.feature_extractor import LandmarkFeatureExtractor, PixelMotionFeatureExtractor
from src.models.classifier import SignClassifier


class VideoSignPredictor:
    """
    Inference engine that loads model weights, applies consistent preprocessing
    (landmark sequence or pixel motion), and returns predictions with uncertainty
    calibration and verified medical meanings.
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
        feat_type = prep_cfg.get("feature_type", "spatio_temporal_pixel_motion")

        if feat_type == "landmark":
            self.extractor = LandmarkFeatureExtractor(
                num_frames=prep_cfg.get("num_frames", 16),
                include_temporal_summary=prep_cfg.get("include_temporal_summary", True),
            )
        else:
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

    def _predict_from_features(self, features: np.ndarray, source_desc: str = "") -> Dict:
        """Helper to run classification and uncertainty gating on extracted features."""
        warnings = []
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

        return {
            "status": "SUCCESS",
            "source": source_desc,
            "predicted_label": top1_label,
            "confidence_score": top1_prob,
            "score_interpretation": "Normalized softmax probability from calibrated classifier",
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

    def predict_video(self, video_path: str) -> Dict:
        """Run inference on a video file."""
        if not os.path.exists(video_path):
            return {
                "status": "ERROR",
                "error": f"File does not exist: {video_path}",
                "is_uncertain": True,
            }

        try:
            features = self.extractor.extract_from_path(video_path)
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Video decoding or feature extraction failed: {str(e)}",
                "is_uncertain": True,
            }

        res = self._predict_from_features(features, source_desc=video_path)
        res["video_path"] = video_path
        return res

    def predict_frames(self, frames: List[np.ndarray]) -> Dict:
        """Run inference on a sequence of image frames (e.g. from live webcam buffer)."""
        if not frames:
            return {
                "status": "ERROR",
                "error": "Empty frame list provided",
                "is_uncertain": True,
            }

        try:
            features = self.extractor.extract_from_frames(frames)
        except Exception as e:
            return {
                "status": "ERROR",
                "error": f"Feature extraction failed on frames: {str(e)}",
                "is_uncertain": True,
            }

        return self._predict_from_features(features, source_desc=f"{len(frames)} frames")
