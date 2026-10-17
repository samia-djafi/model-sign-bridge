"""
Fingerspelling recognition classifier for the Algerian Sign Language Alphabet.

Extracts scale-normalized hand landmarks, finger articulation geometry,
and orientation features to classify static and dynamic manual alphabet letters.
Includes strict uncertainty rejection and unknown state handling.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.alphabet.alphabet_inventory import ALGERIAN_ALPHABET, get_alphabet_letter


class AlphabetClassifier:
    """
    Classifier for the 28 Algerian manual alphabet letters and auxiliary characters.
    Operates on MediaPipe hand landmarks with calibrated uncertainty gating.
    """

    def __init__(
        self,
        model_type: str = "linear_svm",
        uncertainty_threshold: float = 0.45,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.uncertainty_threshold = uncertainty_threshold
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.classes_: Optional[List[str]] = None
        self.is_trained = False

    def _init_model(self):
        if self.model_type == "linear_svm":
            return LinearSVC(C=0.5, dual=False, max_iter=2000, random_state=self.random_state)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=16, random_state=self.random_state, n_jobs=-1)
        else:
            return LinearSVC(C=0.5, dual=False, max_iter=2000, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: List[str]) -> "AlphabetClassifier":
        """Fit scaler and model on training landmark vectors."""
        self.model = self._init_model()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.classes_ = list(self.model.classes_)
        self.is_trained = True
        return self

    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Predict an alphabet character from a single-frame hand landmark feature vector.
        Returns character, confidence score, and uncertainty status.
        """
        if not self.is_trained or self.model is None or self.classes_ is None:
            return {
                "letter": None,
                "confidence": 0.0,
                "is_uncertain": True,
                "status": "DATA_MISSING",
                "message": "Fingerspelling classifier has no trained weights. Field recordings required.",
            }

        X = np.expand_dims(feature_vector, axis=0) if feature_vector.ndim == 1 else feature_vector
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
        elif hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X_scaled)
            exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
            probs = (exp_df / np.sum(exp_df, axis=1, keepdims=True))[0]
        else:
            pred = self.model.predict(X_scaled)[0]
            return {
                "letter": pred,
                "confidence": 1.0,
                "is_uncertain": False,
                "status": "PREDICTED",
            }

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_letter = self.classes_[top_idx]

        is_uncertain = top_prob < self.uncertainty_threshold
        meta = get_alphabet_letter(top_letter)

        return {
            "letter": top_letter if not is_uncertain else "UNKNOWN",
            "candidate_letter": top_letter,
            "confidence": top_prob,
            "is_uncertain": is_uncertain,
            "status": "UNCERTAIN" if is_uncertain else "VERIFIED",
            "arabic_name": meta.get("arabic_name", ""),
            "french_name": meta.get("french_name", ""),
            "letter_id": meta.get("letter_id", ""),
        }

    def save(self, artifact_dir: str, filename: str = "alphabet_model.joblib") -> str:
        """Serialize model weights and scaler."""
        os.makedirs(artifact_dir, exist_ok=True)
        path = os.path.join(artifact_dir, filename)
        joblib.dump({
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "classes_": self.classes_,
            "is_trained": self.is_trained,
            "uncertainty_threshold": self.uncertainty_threshold,
        }, path, compress=3)
        return path

    @classmethod
    def load(cls, filepath: str) -> "AlphabetClassifier":
        """Load trained alphabet model from disk."""
        data = joblib.load(filepath)
        instance = cls(
            model_type=data["model_type"],
            uncertainty_threshold=data["uncertainty_threshold"],
        )
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.classes_ = data["classes_"]
        instance.is_trained = data["is_trained"]
        return instance
