"""
Unit tests for the 100 Practical Healthcare Sentences Library.
"""

import pytest
from src.translation.sentence_library import (
    HEALTHCARE_SENTENCE_LIBRARY,
    get_all_sentences,
    get_sentences_by_category,
    search_sentences,
    get_library_summary,
)


def test_sentence_library_total_count():
    """Verify library contains exactly 100 sentences with unique IDs."""
    sentences = get_all_sentences()
    assert len(sentences) == 100, f"Expected 100 sentences, got {len(sentences)}"

    ids = [s["sentence_id"] for s in sentences]
    assert len(ids) == len(set(ids)), "Duplicate sentence IDs detected!"
    for sid in ids:
        assert sid.startswith("SEN-")


def test_sentence_category_distribution():
    """Verify that all 7 healthcare domains are populated."""
    summary = get_library_summary()
    assert summary["total_sentences"] == 100
    cats = summary["category_distribution"]
    assert len(cats) == 7

    assert cats["Symptoms & Pain"] == 20
    assert cats["Pharmacy Requests"] == 18
    assert cats["Medication & Dosage"] == 16
    assert cats["Emergencies & Urgent Help"] == 16
    assert cats["Allergies & Contraindications"] == 14
    assert cats["Patient States"] == 8
    assert cats["Communication Guidance"] == 8


def test_generation_modes():
    """Verify dual generation modes (phrase_library and dynamic_composition)."""
    summary = get_library_summary()
    modes = summary["generation_modes"]
    assert modes["phrase_library"] == 70
    assert modes["dynamic_composition"] == 30


def test_sentence_fields_and_search():
    """Verify that sentences have valid multilingual text and can be queried."""
    for s in HEALTHCARE_SENTENCE_LIBRARY:
        assert "french_text" in s and len(s["french_text"]) > 5
        assert "arabic_text" in s and len(s["arabic_text"]) > 5
        assert "english_gloss" in s and len(s["english_gloss"]) > 5
        assert "required_tokens" in s and len(s["required_tokens"]) >= 1

    # Test category filtering
    symptom_sentences = get_sentences_by_category("Symptoms & Pain")
    assert len(symptom_sentences) == 20

    # Test search query
    fever_matches = search_sentences("fièvre")
    assert len(fever_matches) > 0

    arabic_matches = search_sentences("طبيب")
    assert len(arabic_matches) > 0
