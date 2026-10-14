"""
Label parser and Unicode normalizer for Algerian Sign Language (VDzSL) dataset.

Handles:
- Unicode NFC normalization
- Trailing and leading whitespace stripping (e.g., 'أمتعة ', 'اسم ')
- Distinction between numeric signs (e.g., '1', '10') and Arabic text signs
- Generation of filesystem-safe ASCII slugs
- Bidirectional mapping between canonical labels and integer class indices
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


class LabelParser:
    """
    Robust label parser ensuring consistent character encoding,
    Unicode normalization, and label-to-index mapping.
    """

    def __init__(self, labels: Optional[List[str]] = None):
        """
        Initialize LabelParser.
        If a list of raw labels is provided, builds the canonical vocabulary.
        """
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.label_metadata: Dict[str, dict] = {}

        if labels:
            self.fit(labels)

    @staticmethod
    def normalize_label(raw_text: str) -> str:
        """
        Normalize label string:
        - Apply Unicode NFC normalization (Standard form C)
        - Strip leading and trailing whitespace (found in original VDzSL files)
        - Collapse multiple internal whitespace characters to a single space
        """
        if not raw_text:
            return ""
        # Unicode NFC normalization
        normalized = unicodedata.normalize("NFC", raw_text)
        # Strip trailing/leading whitespace and collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def extract_label_from_path(file_path: str) -> Tuple[str, str]:
        """
        Extract raw label and normalized canonical label from a file path.

        Returns:
            Tuple of (raw_stem, canonical_label)
        """
        filename = os.path.basename(file_path)
        stem, _ = os.path.splitext(filename)
        canonical = LabelParser.normalize_label(stem)
        return stem, canonical

    @staticmethod
    def label_to_slug(canonical_label: str) -> str:
        """
        Convert a canonical label to a filesystem-safe ASCII slug.
        Numeric signs keep their numbers.
        Arabic signs are represented as hex-unicode or transliterated token.
        """
        if canonical_label.isdigit():
            return f"num_{canonical_label}"

        # Create safe hex representation of Arabic unicode points for cross-platform file paths
        hex_repr = "_".join(f"{ord(c):04x}" for c in canonical_label if not c.isspace())
        return f"ar_{hex_repr}"

    def fit(self, raw_labels: List[str]) -> "LabelParser":
        """
        Build the canonical vocabulary from a collection of raw label strings.
        Preserves deterministic sorting (numerics sorted numerically, Arabic alphabetically).
        """
        canonical_set = set()
        raw_to_canonical = {}

        for raw in raw_labels:
            canon = self.normalize_label(raw)
            if canon:
                canonical_set.add(canon)
                raw_to_canonical[raw] = canon

        # Deterministic sorting:
        # Separate numeric labels from textual labels
        numeric_labels = sorted(
            [l for l in canonical_set if l.isdigit()],
            key=lambda x: int(x)
        )
        text_labels = sorted(
            [l for l in canonical_set if not l.isdigit()]
        )

        sorted_vocabulary = numeric_labels + text_labels

        self.label_to_id = {label: idx for idx, label in enumerate(sorted_vocabulary)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        # Build metadata per class
        for label, idx in self.label_to_id.items():
            self.label_metadata[label] = {
                "class_id": idx,
                "canonical_label": label,
                "is_numeric": label.isdigit(),
                "slug": self.label_to_slug(label),
                "char_count": len(label),
                "word_count": len(label.split()),
            }

        return self

    def encode(self, label_or_path: str) -> int:
        """
        Encode a label or file path to its canonical class integer ID.
        Raises KeyError if label is unknown.
        """
        if os.path.sep in label_or_path or label_or_path.endswith(".mp4"):
            _, canon = self.extract_label_from_path(label_or_path)
        else:
            canon = self.normalize_label(label_or_path)

        if canon not in self.label_to_id:
            raise KeyError(f"Label '{canon}' not found in vocabulary.")
        return self.label_to_id[canon]

    def decode(self, class_id: int) -> str:
        """
        Decode an integer class ID to its canonical label.
        Raises KeyError if ID is unknown.
        """
        if class_id not in self.id_to_label:
            raise KeyError(f"Class ID {class_id} not found in vocabulary.")
        return self.id_to_label[class_id]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    def save(self, filepath: str) -> None:
        """Save label parser mapping to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "num_classes": self.num_classes,
            "label_to_id": self.label_to_id,
            "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
            "metadata": self.label_metadata,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LabelParser":
        """Load label parser mapping from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        parser = cls()
        parser.label_to_id = payload["label_to_id"]
        parser.id_to_label = {int(k): v for k, v in payload["id_to_label"].items()}
        parser.label_metadata = payload.get("metadata", {})
        return parser
