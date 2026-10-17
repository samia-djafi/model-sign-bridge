"""
Unit tests for Algerian manual alphabet inventory, fingerspelling state machine, and classifier.
"""

import time
import pytest
from src.alphabet.alphabet_inventory import ALGERIAN_ALPHABET
from src.alphabet.fingerspelling_buffer import FingerspellingBuffer
from src.alphabet.alphabet_classifier import AlphabetClassifier


def test_alphabet_inventory_size_and_keys():
    """Verify inventory contains exactly 31 verified Algerian manual alphabet letters."""
    assert len(ALGERIAN_ALPHABET) == 31, f"Expected 31 letters, got {len(ALGERIAN_ALPHABET)}"
    expected_auxiliary = ["ة", "لا", "ء"]
    for aux in expected_auxiliary:
        assert aux in ALGERIAN_ALPHABET, f"Missing auxiliary sign '{aux}'"


def test_alphabet_schema_and_uniqueness():
    """Verify all letters have unique IDs and required metadata fields."""
    ids = [entry["letter_id"] for entry in ALGERIAN_ALPHABET.values()]
    assert len(ids) == len(set(ids)), "Duplicate letter IDs detected!"

    for char, info in ALGERIAN_ALPHABET.items():
        assert info["letter_id"].startswith("ALPHA-")
        assert info["arabic_char"] == char
        assert len(info["arabic_name"]) > 0
        assert len(info["french_name"]) > 0
        assert info["handshape_type"] in ["static", "dynamic"]
        assert info["status"] in ["reference_verified", "data_missing", "data_available", "training_complete"]


def test_fingerspelling_buffer_typing_and_backspace():
    """Test interactive typing, debouncing, backspacing, and space insertion."""
    buf = FingerspellingBuffer(debounce_seconds=0.05)
    assert buf.current_text == ""

    # Add characters
    assert buf.add_letter("أ") is True
    assert buf.current_text == "أ"

    # Rapid repeated character within debounce window should be ignored
    assert buf.add_letter("أ") is False
    assert buf.current_text == "أ"

    time.sleep(0.06)
    assert buf.add_letter("س") is True
    assert buf.current_text == "أس"

    time.sleep(0.06)
    assert buf.add_letter("د") is True
    assert buf.current_text == "أسد"

    # Backspace
    assert buf.backspace() is True
    assert buf.current_text == "أس"

    # Space
    buf.add_space()
    assert buf.current_text == "أس"

    # Add another word
    time.sleep(0.06)
    buf.add_letter("ب")
    assert buf.current_text == "أس ب"

    # Word finalize
    final_word = buf.finalize()
    assert "أس ب" in final_word
    assert buf.current_text == ""


def test_fingerspelling_transliteration():
    """Verify French phonetic transliteration."""
    buf = FingerspellingBuffer(debounce_seconds=0.01)
    buf.add_letter("س")
    time.sleep(0.02)
    buf.add_letter("ل")
    time.sleep(0.02)
    buf.add_letter("م")
    translit = buf.get_transliteration()
    assert "S" in translit
    assert "L" in translit
    assert "M" in translit


def test_alphabet_classifier_uncertainty():
    """Test classifier wrapper returns UNKNOWN when confidence is below threshold."""
    import numpy as np
    clf = AlphabetClassifier(uncertainty_threshold=0.6)
    res = clf.predict(np.zeros(63))
    assert res["is_uncertain"] is True
    assert res["confidence"] == 0.0
    assert res["status"] == "DATA_MISSING"
