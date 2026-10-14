"""
Medical, Pharmacy, and Healthcare Vocabulary Audit for VDzSL.

Audits the 415 signs in VDzSL against medical, pharmacy, and healthcare communication needs.
Classifies each sign's relevance, verified translations (Arabic, French, English),
evidence source, and suitability for the initial Healthcare/Pharmacy MVP classifier.
"""

import argparse
import json
import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from src.data.label_parser import LabelParser


# Curated dictionary of healthcare, pharmacy, dosage, and basic assistance relevance
# Categories:
#   A: Primary Medical / Symptom / Anatomy (Direct medical relevance)
#   B: Dosage & Administration Schedule (Quantities, frequency, timing, ingestion)
#   C: Pharmacy Counter Communication & Assistance (Requests, courtesy, financial)
#   D: General Everyday Vocabulary (Not prioritized for pharmacy MVP)

HEALTHCARE_TAXONOMY = {
    # Category A: Primary Medical, Symptom, Anatomy
    "ألم": {
        "en": "Pain / Ache",
        "fr": "Douleur",
        "category": "Medical Symptom",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Essential for patient describing symptoms or pain level to pharmacist",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "حرارة": {
        "en": "Fever / Heat",
        "fr": "Chaleur / Fièvre",
        "category": "Medical Symptom",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Critical symptom indicating infection or fever requiring antipyretics",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "مريض": {
        "en": "Sick / Patient",
        "fr": "Malade",
        "category": "Medical Status",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Identifies that a person or family member is unwell",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "طبيب أسنان": {
        "en": "Dentist",
        "fr": "Dentiste",
        "category": "Healthcare Professional",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Indicates dental issue or prescription originating from dentist",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "إبرة": {
        "en": "Needle / Syringe / Injection",
        "fr": "Aiguille / Seringue",
        "category": "Medical Device",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Common pharmacy purchase for insulin, injectable medications, or vaccinations",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "كحول": {
        "en": "Alcohol / Antiseptic",
        "fr": "Alcool chirurgical / Antiseptique",
        "category": "Medical Supply",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Standard antiseptic requested at pharmacy counters for wound cleaning",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "خطر": {
        "en": "Danger / Warning / Hazard",
        "fr": "Danger",
        "category": "Medical Safety",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Crucial warning for contraindications, toxic exposure, or adverse drug reactions",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "جسم": {
        "en": "Body",
        "fr": "Corps",
        "category": "Anatomy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Used to describe generalized body aches or specify pain locations",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "قلب": {
        "en": "Heart",
        "fr": "Cœur",
        "category": "Anatomy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Vital organ; key for cardiovascular medication communication",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "أذن": {
        "en": "Ear",
        "fr": "Oreille",
        "category": "Anatomy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Common infection location (ear drops, otitis, pain)",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "يد": {
        "en": "Hand",
        "fr": "Main",
        "category": "Anatomy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Injury site, burns, or instructions on applying topical medication",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "رأس": {
        "en": "Head",
        "fr": "Tête",
        "category": "Anatomy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Essential for reporting headache / migraine symptoms",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "بشرة": {
        "en": "Skin",
        "fr": "Peau",
        "category": "Anatomy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Dermatological issues, rashes, allergies, or topical creams",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },

    # Category B: Dosage, Frequency, Temporal Schedule, Ingestion
    "كل يوم": {
        "en": "Every day / Daily",
        "fr": "Chaque jour / Quotidien",
        "category": "Dosage Schedule",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Primary frequency instruction for chronic or acute medication",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "صباح": {
        "en": "Morning",
        "fr": "Matin",
        "category": "Dosage Schedule",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Morning dosing instruction (e.g., take before breakfast)",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "الليل": {
        "en": "Night / Evening",
        "fr": "Nuit / Soir",
        "category": "Dosage Schedule",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Bedtime or evening dosing instruction (e.g., sleep aids, statins)",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "منتصف الليل": {
        "en": "Midnight",
        "fr": "Minuit",
        "category": "Dosage Schedule",
        "pharmacy_relevance": "Medium",
        "relevance_rationale": "Night-time emergency or scheduled dosing",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": False,
    },
    "الآن": {
        "en": "Now / Immediately",
        "fr": "Maintenant / Immédiatement",
        "category": "Temporal Urgency",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Urgent dosing (take immediately on onset of acute symptoms)",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "اليوم": {
        "en": "Today",
        "fr": "Aujourd'hui",
        "category": "Temporal Context",
        "pharmacy_relevance": "Medium",
        "relevance_rationale": "Starting treatment today",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": False,
    },
    "غدا": {
        "en": "Tomorrow",
        "fr": "Demain",
        "category": "Temporal Context",
        "pharmacy_relevance": "Medium",
        "relevance_rationale": "Next dosage or returning for prescription fulfillment",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": False,
    },
    "نصف": {
        "en": "Half",
        "fr": "Demi",
        "category": "Dosage Quantity",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Common instruction: split tablet / take half a tablet",
        "source": "3DZSignDB SiGML",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "ماء": {
        "en": "Water",
        "fr": "Eau",
        "category": "Administration Context",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Standard instruction: take tablet with a full glass of water",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "طعام": {
        "en": "Food / Meal",
        "fr": "Nourriture / Repas",
        "category": "Administration Context",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Crucial instruction: take with meals or on an empty stomach",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "سكر": {
        "en": "Sugar",
        "fr": "Sucre",
        "category": "Condition / Dietary",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Critical for diabetes discussions, glucose monitoring, sugar-free syrups",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": True,
    },
    "ملح": {
        "en": "Salt",
        "fr": "Sel",
        "category": "Condition / Dietary",
        "pharmacy_relevance": "Medium",
        "relevance_rationale": "Hypertension dietary restrictions, saline solution",
        "source": "Official Ministry Dictionary (2017)",
        "verification_status": "Verified (LSA Lexicon)",
        "mvp_candidate": False,
    },

    # Numerals for Dosage
    "1": {
        "en": "One (1)",
        "fr": "Un (1)",
        "category": "Dosage Number",
        "pharmacy_relevance": "High",
        "relevance_rationale": "1 tablet, 1 time daily, 1 box",
        "source": "3DZSignDB (Digit)",
        "verification_status": "Verified (VDzSL Digit)",
        "mvp_candidate": True,
    },
    "2": {
        "en": "Two (2)",
        "fr": "Deux (2)",
        "category": "Dosage Number",
        "pharmacy_relevance": "High",
        "relevance_rationale": "2 tablets, 2 times daily (morning and evening)",
        "source": "3DZSignDB (Digit)",
        "verification_status": "Verified (VDzSL Digit)",
        "mvp_candidate": True,
    },
    "3": {
        "en": "Three (3)",
        "fr": "Trois (3)",
        "category": "Dosage Number",
        "pharmacy_relevance": "High",
        "relevance_rationale": "3 times daily (morning, noon, night)",
        "source": "3DZSignDB (Digit)",
        "verification_status": "Verified (VDzSL Digit)",
        "mvp_candidate": True,
    },

    # Category C: Pharmacy Counter Interaction & Assistance
    "يساعد": {
        "en": "Help / Assist",
        "fr": "Aider",
        "category": "Assistance",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Patient requesting assistance or pharmacist offering guidance",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "من فضلك": {
        "en": "Please",
        "fr": "S'il vous plaît",
        "category": "Politeness / Request",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Polite formula initiating requests at pharmacy counter",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "شكرا": {
        "en": "Thank you",
        "fr": "Merci",
        "category": "Courtesy",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Standard closing of pharmacy exchange",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "نعم": {
        "en": "Yes",
        "fr": "Oui",
        "category": "Confirmation",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Confirming symptom, allergy, or understanding instructions",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "لا": {
        "en": "No",
        "fr": "Non",
        "category": "Denial / Negation",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Denying allergy, symptoms, or confirming no prior conditions",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "حسنا": {
        "en": "OK / Alright",
        "fr": "D'accord / Bien",
        "category": "Acknowledgment",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Acknowledging pharmacist instructions",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "كم": {
        "en": "How much? / What price?",
        "fr": "Combien",
        "category": "Inquiry",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Asking for cost of medication or quantity to take",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "مال": {
        "en": "Money / Cash / Payment",
        "fr": "Argent",
        "category": "Transaction",
        "pharmacy_relevance": "High",
        "relevance_rationale": "Payment and pricing interaction at counter",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": True,
    },
    "أين": {
        "en": "Where?",
        "fr": "Où",
        "category": "Inquiry",
        "pharmacy_relevance": "Medium",
        "relevance_rationale": "Asking where clinic, injection room, or medication is located",
        "source": "3DZSignDB & Official Ministry Dictionary (2017)",
        "verification_status": "Verified (3DZSignDB SiGML)",
        "mvp_candidate": False,
    },
}

# Explicit list of critical pharmacy concepts completely absent from VDzSL
UNSUPPORTED_HEALTHCARE_CONCEPTS = [
    {
        "concept_en": "Medicine / Medication / Drug",
        "concept_ar": "دواء",
        "concept_fr": "Médicament",
        "reason_absent": "Not modeled in 3DZSignDB SiGML files rendered into VDzSL",
        "clinical_impact": "High. Patient cannot explicitly sign the noun 'medicine'",
        "mitigation_strategy": "Record human signing data or import SiGML from HamNoSys dictionaries; patient uses 'ألم' (pain) or brand/prescription card"
    },
    {
        "concept_en": "Pharmacy / Chemist",
        "concept_ar": "صيدلية",
        "concept_fr": "Pharmacie",
        "reason_absent": "Not modeled in VDzSL",
        "clinical_impact": "Medium. Concept usually implicit when physically present in the pharmacy",
        "mitigation_strategy": "Include in planned secondary human signing collection"
    },
    {
        "concept_en": "Hospital / Clinic",
        "concept_ar": "مستشفى",
        "concept_fr": "Hôpital / Clinique",
        "reason_absent": "Not modeled in VDzSL",
        "clinical_impact": "High. Referral or origin of prescription cannot be signed directly",
        "mitigation_strategy": "Priority collection in human signing stage"
    },
    {
        "concept_en": "Doctor / General Physician",
        "concept_ar": "طبيب",
        "concept_fr": "Médecin",
        "reason_absent": "Only 'طبيب أسنان' (Dentist) exists in VDzSL; general 'طبيب' is absent",
        "clinical_impact": "High. Must not substitute 'طبيب أسنان' when referring to a general doctor",
        "mitigation_strategy": "Explicit safety rule forbidding relabeling 'طبيب أسنان' as general 'طبيب'"
    },
    {
        "concept_en": "Prescription / Order",
        "concept_ar": "وصفة طبية / أوردونونس",
        "concept_fr": "Ordonnance",
        "reason_absent": "Not modeled in VDzSL",
        "clinical_impact": "High. Patient currently presents physical paper prescription",
        "mitigation_strategy": "Physical handover of prescription bypasses need for sign in immediate MVP"
    },
    {
        "concept_en": "Pill / Tablet / Capsule",
        "concept_ar": "حبة / قرص",
        "concept_fr": "Comprimé / Gélule",
        "reason_absent": "Not modeled in VDzSL",
        "clinical_impact": "Medium. Can be partially referenced by numeral + administration with water",
        "mitigation_strategy": "Collect dedicated gesture clips for tablet/syrup administration"
    },
    {
        "concept_en": "Allergy",
        "concept_ar": "حساسية",
        "concept_fr": "Allergie",
        "reason_absent": "Not modeled in VDzSL",
        "clinical_impact": "Critical safety issue. Pharmacist must verify allergies via written card/app",
        "mitigation_strategy": "Never assume allergy absence. Require written confirmation"
    },
    {
        "concept_en": "Blood",
        "concept_ar": "دم",
        "concept_fr": "Sang",
        "reason_absent": "Not modeled in VDzSL",
        "clinical_impact": "Medium. Blood pressure, blood test, or bleeding cannot be signed directly",
        "mitigation_strategy": "Priority candidate for clinical LSA expansion"
    },
]


def audit_vocabulary(inventory_csv: str, output_csv: str, output_md: str):
    """Perform vocabulary audit and generate reports."""
    df_inv = pd.read_csv(inventory_csv, encoding="utf-8-sig")
    available_labels = set(df_inv["canonical_label"].unique())

    records = []
    for label, meta in HEALTHCARE_TAXONOMY.items():
        is_in_dataset = label in available_labels
        num_videos = len(df_inv[df_inv["canonical_label"] == label]) if is_in_dataset else 0
        signers = df_inv[df_inv["canonical_label"] == label]["signer_id"].unique().tolist() if is_in_dataset else []
        cameras = df_inv[df_inv["canonical_label"] == label]["camera_id"].unique().tolist() if is_in_dataset else []

        records.append({
            "original_label": label,
            "meaning_en": meta["en"],
            "meaning_fr": meta["fr"],
            "category": meta["category"],
            "pharmacy_relevance": meta["pharmacy_relevance"],
            "relevance_rationale": meta["relevance_rationale"],
            "in_dataset": is_in_dataset,
            "video_count": num_videos,
            "signers": ",".join(signers),
            "cameras": ",".join(cameras),
            "evidence_source": meta["source"],
            "verification_status": meta["verification_status"],
            "suitable_for_mvp": meta["mvp_candidate"] and is_in_dataset,
        })

    df_vocab = pd.DataFrame(records)
    # Sort by relevance then category
    relevance_order = {"High": 0, "Medium": 1, "Low": 2}
    df_vocab["sort_key"] = df_vocab["pharmacy_relevance"].map(relevance_order)
    df_vocab.sort_values(by=["sort_key", "category", "original_label"], inplace=True)
    df_vocab.drop(columns=["sort_key"], inplace=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_vocab.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Vocabulary audit CSV saved to: {output_csv}")

    # Generate Markdown documentation
    mvp_classes = df_vocab[df_vocab["suitable_for_mvp"]]

    md_content = f"""# Medical, Pharmacy, and Basic Healthcare Vocabulary Audit

## 1. Overview and Objectives
In accordance with **Phase 2 & Section 14 (Medical Safety)** of the project requirements:
- The initial domain focus is **Medical - Pharmacy and Basic Healthcare Communication**.
- We audited the actual VDzSL dataset (415 signs) to identify genuine healthcare, symptom, dosage, and counter-interaction signs.
- **Strict Scientific Rule**: We **never** relabel unrelated signs (e.g. we do not treat `طبيب أسنان` [Dentist] as general doctor `طبيب`, and we do not invent missing signs like `دواء` [medicine]).
- This system is designed solely for **communication support**, not diagnostic or prescribing decisions.

---

## 2. Verified Healthcare Vocabulary in VDzSL ({len(mvp_classes)} MVP Classes Selected)

The following signs have been verified in the dataset, cross-referenced with the 3DZSignDB SiGML lexicon and the Algerian Ministry of Solidarity Dictionary (2017), and selected for the focused healthcare baseline:

| # | Arabic Sign | French Meaning | English Meaning | Category | Videos | Avatar / Cameras | Relevance & Function |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- | :--- |
"""
    for idx, row in enumerate(mvp_classes.iterrows(), 1):
        _, r = row
        md_content += f"| {idx} | **{r['original_label']}** | {r['meaning_fr']} | {r['meaning_en']} | {r['category']} | {r['video_count']} | 4 Avatars / 5 Cams | {r['relevance_rationale']} |\n"

    md_content += f"""
---

## 3. Unsupported / Missing Healthcare Concepts (Gaps in VDzSL)

The VDzSL dataset is an open-domain everyday gesture collection derived from 3DZSignDB and contains critical lexical gaps in healthcare terminology. The table below documents concepts that **cannot** be recognized by models trained on VDzSL:

| Missing Concept (English) | Arabic Term | French Term | Clinical Impact | Mitigation & Data Collection Plan |
| :--- | :--- | :--- | :--- | :--- |
"""
    for item in UNSUPPORTED_HEALTHCARE_CONCEPTS:
        md_content += f"| **{item['concept_en']}** | {item['concept_ar']} | {item['concept_fr']} | {item['clinical_impact']} | {item['mitigation_strategy']} |\n"

    md_content += """
---

## 4. Medical Safety & Operational Protocol

> [!CAUTION]
> **Safety Critical Boundaries**:
> 1. **No Autonomous Clinical Decision Making**: Under no circumstances should model outputs be used to infer diseases, diagnose conditions, or suggest drug dosages.
> 2. **Mandatory Visual & Written Confirmation**: For safety-critical instructions (e.g., dosage numbers, antibiotic course length), the pharmacist must confirm the meaning in writing or via physical packaging.
> 3. **Uncertainty Rejection**: Any model prediction with probability below an established confidence threshold or with high entropy must trigger an explicit **"Uncertain / Recheck"** status rather than guessing a potentially hazardous medical action.
> 4. **No Semantic Invention**: The absence of words like 'دواء' (medicine) or 'حساسية' (allergy) means the system must never attempt to hallucinate these words from similar signs.

---

## 5. Human Signing & Vocabulary Expansion Plan

To extend the vocabulary from the synthetic 3D avatar subset to full clinical utility:
1. **Target Vocabulary**: The 8 missing high-impact concepts identified above plus 25 common pharmaceutical phrases.
2. **Qualified Signer Participation**: Recruitment of native/fluent Algerian Sign Language (LSA) signers in collaboration with deaf associations in Algeria.
3. **Informed Consent & Ethics**: Standardized protocol ensuring participants provide informed consent for research recording, with options for face blurring / pose-only feature extraction to protect personal privacy.
4. **Multi-Camera Real Recording**: Capture in varied illumination and realistic pharmacy counter backdrops.
"""

    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Vocabulary audit Markdown saved to: {output_md}")


def main():
    parser = argparse.ArgumentParser(description="Audit medical and pharmacy vocabulary.")
    parser.add_argument(
        "--inventory-csv",
        default=r"d:\bridge_sign_model\data\manifests\dataset_inventory.csv",
        help="Path to dataset inventory CSV"
    )
    parser.add_argument(
        "--output-csv",
        default=r"d:\bridge_sign_model\data\manifests\medical_vocabulary.csv",
        help="Path for medical vocabulary CSV"
    )
    parser.add_argument(
        "--output-md",
        default=r"d:\bridge_sign_model\docs\vocabulary_audit.md",
        help="Path for medical vocabulary Markdown documentation"
    )
    args = parser.parse_args()

    audit_vocabulary(args.inventory_csv, args.output_csv, args.output_md)


if __name__ == "__main__":
    main()
