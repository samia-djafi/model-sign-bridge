"""
Unit tests for the 150 Target Vocabulary Concepts Inventory.
"""

import pytest
from src.data.vocabulary_inventory import (
    MEDICAL_PHARMACY_100,
    EMERGENCY_30,
    GENERAL_HEALTHCARE_20,
    TARGET_VOCABULARY_150,
    get_inventory_summary,
)

VALID_STATUSES = {
    "training_complete",
    "data_available",
    "reference_verified",
    "data_missing",
    "signer_validated",
}


def test_vocabulary_counts():
    """Verify exact counts of target concepts across all categories."""
    assert len(MEDICAL_PHARMACY_100) == 100, f"Expected 100 MED, got {len(MEDICAL_PHARMACY_100)}"
    assert len(EMERGENCY_30) == 30, f"Expected 30 EMG, got {len(EMERGENCY_30)}"
    assert len(GENERAL_HEALTHCARE_20) == 20, f"Expected 20 GEN, got {len(GENERAL_HEALTHCARE_20)}"
    assert len(TARGET_VOCABULARY_150) == 150, f"Expected 150 total, got {len(TARGET_VOCABULARY_150)}"


def test_vocabulary_id_uniqueness():
    """Verify that every concept ID is strictly unique and correctly prefixed."""
    ids = [c["id"] for c in TARGET_VOCABULARY_150]
    assert len(ids) == len(set(ids)), "Duplicate concept IDs detected!"

    med_ids = [c["id"] for c in MEDICAL_PHARMACY_100]
    for mid in med_ids:
        assert mid.startswith("MED-")

    emg_ids = [c["id"] for c in EMERGENCY_30]
    for eid in emg_ids:
        assert eid.startswith("EMG-")

    gen_ids = [c["id"] for c in GENERAL_HEALTHCARE_20]
    for gid in gen_ids:
        assert gid.startswith("GEN-")


def test_vocabulary_schema_and_statuses():
    """Verify that all concept entries have complete multilingual labels and valid statuses."""
    for c in TARGET_VOCABULARY_150:
        assert "id" in c and len(c["id"]) > 0
        assert "ar" in c and len(c["ar"]) > 0
        assert "fr" in c and len(c["fr"]) > 0
        assert "en" in c and len(c["en"]) > 0
        assert "cat" in c and len(c["cat"]) > 0
        assert "ref" in c and len(c["ref"]) > 0
        assert "status" in c and c["status"] in VALID_STATUSES
        assert "clips" in c and isinstance(c["clips"], int)


def test_inventory_summary():
    """Verify structured summary output matches authoritative numbers."""
    summary = get_inventory_summary()
    assert summary["total_target_concepts"] == 150
    assert summary["category_breakdown"]["Medical & Pharmacy"] == 100
    assert summary["category_breakdown"]["Emergency"] == 30
    assert summary["category_breakdown"]["General Healthcare"] == 20
    assert sum(summary["status_counts"].values()) == 150
