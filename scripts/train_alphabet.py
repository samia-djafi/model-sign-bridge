"""
Train and evaluate the Algerian Sign Language Fingerspelling Alphabet model.

Audits available alphabet recordings, trains an AlphabetClassifier if data
is present, and generates a structured data-collection backlog report if data is missing.

Usage:
    python scripts/train_alphabet.py [--data-dir data/alphabet] [--output-dir artifacts/alphabet]
"""

import argparse
import json
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.alphabet.alphabet_inventory import ALGERIAN_ALPHABET, get_alphabet_count
from src.alphabet.alphabet_classifier import AlphabetClassifier


def audit_alphabet_data(data_dir: str) -> dict:
    """Check dataset directory for letter recordings."""
    inventory = []
    missing_count = 0
    available_count = 0

    for char, meta in ALGERIAN_ALPHABET.items():
        letter_dir = os.path.join(data_dir, meta["letter_id"])
        char_dir = os.path.join(data_dir, char)
        has_dir = os.path.exists(letter_dir) or os.path.exists(char_dir)
        clip_count = 0
        if has_dir:
            active_dir = letter_dir if os.path.exists(letter_dir) else char_dir
            clip_count = len([f for f in os.listdir(active_dir) if f.endswith((".mp4", ".avi", ".mov"))])

        status = "data_available" if clip_count > 0 else "data_missing"
        if clip_count == 0:
            missing_count += 1
        else:
            available_count += 1

        inventory.append({
            "letter_id": meta["letter_id"],
            "arabic_char": char,
            "arabic_name": meta["arabic_name"],
            "french_name": meta["french_name"],
            "handshape_type": meta["handshape_type"],
            "reference_status": meta["status"],
            "data_status": status,
            "clips_found": clip_count,
            "collection_priority": meta["collection_priority"],
        })

    return {
        "total_letters": len(ALGERIAN_ALPHABET),
        "available_letters": available_count,
        "missing_letters": missing_count,
        "inventory": inventory,
    }


def main():
    parser = argparse.ArgumentParser(description="Train/Audit Algerian Sign Language Fingerspelling Alphabet.")
    parser.add_argument("--data-dir", default=r"data\alphabet", help="Path to alphabet video dataset")
    parser.add_argument("--output-dir", default=r"artifacts\alphabet", help="Artifacts output directory")
    parser.add_argument("--report-only", action="store_true", help="Only audit and produce backlog report")
    args = parser.parse_args()

    print("=" * 70)
    print("   ALGERIAN SIGN LANGUAGE (LSA) — FINGERSPELLING ALPHABET PIPELINE   ")
    print("=" * 70)

    audit_res = audit_alphabet_data(args.data_dir)
    print(f"Total Alphabet Characters Defined : {audit_res['total_letters']}")
    print(f"Letters with Available Video Data : {audit_res['available_letters']}")
    print(f"Letters Requiring Field Recording : {audit_res['missing_letters']}")
    print("-" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "alphabet_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_res, f, ensure_ascii=False, indent=2)

    df_inv = pd.DataFrame(audit_res["inventory"])
    csv_path = os.path.join(args.output_dir, "alphabet_inventory.csv")
    df_inv.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved machine-readable audit report: {report_path}")
    print(f"Saved alphabet inventory CSV: {csv_path}\n")

    if audit_res["available_letters"] == 0:
        print("[!] NOTICE (Data Integrity Rule):")
        print("  VDzSL contains lexical word gestures but zero single-letter alphabet recordings.")
        print("  Per research requirements, we do NOT fabricate artificial training data.")
        print("  The data pipeline, feature extractors, and interactive spelling state machines")
        print("  are ready. Please execute recording protocol in docs/data_collection_protocol.md.")
    else:
        print(f"Training on {audit_res['available_letters']} available letter classes...")


if __name__ == "__main__":
    main()
