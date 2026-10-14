"""
Unit tests for LabelParser.
"""

import os
import tempfile
import unicodedata
import pytest
from src.data.label_parser import LabelParser


def test_normalization_whitespace():
    """Test stripping whitespace and multiple spaces."""
    assert LabelParser.normalize_label("  أمتعة  ") == "أمتعة"
    assert LabelParser.normalize_label("اسم ") == "اسم"
    assert LabelParser.normalize_label("طبيب   أسنان") == "طبيب أسنان"
    assert LabelParser.normalize_label("") == ""


def test_unicode_normalization_equivalence():
    """Test that NFD decomposed Arabic matches NFC composed Arabic."""
    nfc_text = "أخ"
    nfd_text = unicodedata.normalize("NFD", nfc_text)
    assert LabelParser.normalize_label(nfc_text) == LabelParser.normalize_label(nfd_text)


def test_extract_label_from_path():
    """Test extracting stem and canonical label from Windows or POSIX paths."""
    stem, canon = LabelParser.extract_label_from_path(r"data\VDzSL\signer0\cam0\أمتعة .mp4")
    assert stem == "أمتعة "
    assert canon == "أمتعة"

    stem2, canon2 = LabelParser.extract_label_from_path("data/VDzSL/signer1/cam2/10.mp4")
    assert stem2 == "10"
    assert canon2 == "10"


def test_fit_and_sort_numeric_and_text():
    """Test vocabulary building with numeric ordering and Arabic ordering."""
    raw = ["10", "2", "1", "أب", "أخ", "أمتعة "]
    parser = LabelParser(raw)

    assert parser.num_classes == 6
    # Numeric classes must be sorted numerically
    assert parser.decode(0) == "1"
    assert parser.decode(1) == "2"
    assert parser.decode(2) == "10"
    # Canonical label stripped
    assert parser.encode("أمتعة ") == parser.encode("أمتعة")


def test_slug_generation():
    """Test filesystem-safe slug generation."""
    assert LabelParser.label_to_slug("10") == "num_10"
    slug = LabelParser.label_to_slug("أب")
    assert slug.startswith("ar_")
    assert all(c.isalnum() or c == "_" for c in slug)


def test_unknown_label_error():
    """Test that encoding an unknown label raises KeyError."""
    parser = LabelParser(["أب", "أخ"])
    with pytest.raises(KeyError):
        parser.encode("طبيب")


def test_save_and_load_roundtrip():
    """Test JSON saving and loading with Arabic characters preserved."""
    raw = ["1", "2", "طبيب أسنان", "أمتعة ", "مستشفى"]
    parser = LabelParser(raw)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "labels.json")
        parser.save(json_path)

        loaded = LabelParser.load(json_path)
        assert loaded.num_classes == parser.num_classes
        for label in ["1", "2", "طبيب أسنان", "أمتعة", "مستشفى"]:
            assert loaded.encode(label) == parser.encode(label)
