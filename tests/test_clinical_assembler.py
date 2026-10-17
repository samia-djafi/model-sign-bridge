"""Unit tests for the ClinicalAssembler translation engine."""

import pytest
from src.translation.clinical_assembler import ClinicalAssembler, SignToken, AssemblyResult


def test_empty_tokens():
    assembler = ClinicalAssembler()
    result = assembler.assemble()
    assert result.raw_tokens == []
    assert "Aucun" in result.french_sentence
    assert result.urgency_level == "NORMAL"


def test_debouncing_consecutive_duplicates():
    assembler = ClinicalAssembler(debounce_window_sec=1.5)
    # Add repeated sign at t=0, t=0.5, t=1.0, t=2.0
    added1 = assembler.add_token("ألم", timestamp=0.0)
    added2 = assembler.add_token("ألم", timestamp=0.5)
    added3 = assembler.add_token("ألم", timestamp=1.0)
    added4 = assembler.add_token("ألم", timestamp=2.0)

    assert added1 is True
    assert added2 is False  # skipped
    assert added3 is False  # skipped
    assert added4 is True   # allowed (diff > 1.5s)
    assert len(assembler.tokens) == 2


def test_symptom_headache_assembly():
    assembler = ClinicalAssembler()
    assembler.add_token("من فضلك")
    assembler.add_token("مريض")
    assembler.add_token("ألم")
    assembler.add_token("رأس")
    assembler.add_token("صباح")

    res = assembler.assemble()
    assert "céphalée" in res.french_sentence.lower() or "tête" in res.french_sentence.lower()
    assert "صداع" in res.arabic_sentence or "الرأس" in res.arabic_sentence
    assert res.clinical_summary["patient_state"] == "Malade / Souffrant"
    assert "Douleur" in res.clinical_summary["symptomes"]
    assert "tête" in res.clinical_summary["siege_anatomique"]
    assert "matin" in res.clinical_summary["temporalite"]


def test_dosage_and_modality_assembly():
    assembler = ClinicalAssembler()
    assembler.add_token("1")
    assembler.add_token("نصف")
    assembler.add_token("كل يوم")
    assembler.add_token("ماء")
    assembler.add_token("طعام")

    res = assembler.assemble()
    assert res.clinical_summary["posologie"]["quantite"] == 1.5
    assert res.clinical_summary["posologie"]["frequence"] == "Chaque jour"
    assert res.clinical_summary["posologie"]["avec_eau"] is True
    assert res.clinical_summary["posologie"]["avec_repas"] is True
    assert "eau" in res.french_sentence.lower()
    assert "الماء" in res.arabic_sentence


def test_emergency_detection():
    assembler = ClinicalAssembler()
    assembler.add_token("خطر")
    assembler.add_token("حرارة")
    assembler.add_token("الآن")

    res = assembler.assemble()
    assert res.urgency_level == "CRITICAL"
    assert "URGENCE" in res.french_sentence
    assert "طارئة" in res.arabic_sentence
