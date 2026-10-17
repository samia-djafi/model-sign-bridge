"""Clinical Assembler: Converts sequences of detected signs into natural sentences and pharmacy records.

In Algerian Sign Language (LSA / ALGSL), signs are not signed as spoken word-for-word sentences.
They represent visual concepts (Glosses). A patient at a pharmacy counter signing:
  [من فضلك, مريض, ألم, رأس, صباح]
is expressing:
  "S'il vous plaît, je suis malade et j'ai mal à la tête depuis ce matin."
  "من فضلك، أنا مريض وأعاني من ألم في الرأس منذ الصباح."

This module implements a clinical slot-filling and rule-based assembler that:
1. Debounces and validates streams of recognized signs.
2. Organizes them into clinical slots: Symptoms, Anatomy, Timing, Dosage, Modality, Urgency.
3. Generates natural sentences in French and Modern Standard Arabic.
4. Produces a structured Clinical Pharmacy Record (Fiche Comptoir Pharmacie).
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


# Linguistic and clinical metadata for all 32 MVP healthcare signs
SIGN_METADATA: Dict[str, Dict[str, Any]] = {
    "ألم": {
        "fr": "douleur",
        "en": "pain",
        "category": "symptom",
        "clinical_term_fr": "douleur / algie",
    },
    "حرارة": {
        "fr": "fièvre",
        "en": "fever",
        "category": "symptom",
        "clinical_term_fr": "hyperthermie / fièvre",
    },
    "مريض": {
        "fr": "malade",
        "en": "sick",
        "category": "state",
        "clinical_term_fr": "état pathologique",
    },
    "خطر": {
        "fr": "danger / urgence",
        "en": "danger / emergency",
        "category": "urgency",
        "clinical_term_fr": "urgence vitale",
    },
    "رأس": {
        "fr": "tête",
        "en": "head",
        "category": "anatomy",
        "clinical_compound_fr": "céphalée (maux de tête)",
        "clinical_compound_ar": "صداع / ألم الرأس",
    },
    "يد": {
        "fr": "main",
        "en": "hand",
        "category": "anatomy",
        "clinical_compound_fr": "douleur / lésion de la main",
        "clinical_compound_ar": "ألم / إصابة في اليد",
    },
    "أذن": {
        "fr": "oreille",
        "en": "ear",
        "category": "anatomy",
        "clinical_compound_fr": "otalgie (douleur d'oreille)",
        "clinical_compound_ar": "ألم الأذن",
    },
    "قلb": {  # fallback
        "fr": "cœur",
        "en": "heart",
        "category": "anatomy",
    },
    "قلب": {
        "fr": "cœur",
        "en": "heart",
        "category": "anatomy",
        "clinical_compound_fr": "douleur précordiale / palpitations",
        "clinical_compound_ar": "ألم في الصدر / القلب",
    },
    "بشرة": {
        "fr": "peau",
        "en": "skin",
        "category": "anatomy",
        "clinical_compound_fr": "affection cutanée / dermatose",
        "clinical_compound_ar": "مشكلة جلدية / حساسية البشرة",
    },
    "جسم": {
        "fr": "corps",
        "en": "body",
        "category": "anatomy",
        "clinical_compound_fr": "douleurs corporelles / courbatures",
        "clinical_compound_ar": "آلام عامة في الجسم",
    },
    "الآن": {
        "fr": "maintenant",
        "en": "now",
        "category": "temporal",
    },
    "صباح": {
        "fr": "matin",
        "en": "morning",
        "category": "temporal",
    },
    "الليل": {
        "fr": "nuit",
        "en": "night",
        "category": "temporal",
    },
    "كل يوم": {
        "fr": "chaque jour",
        "en": "every day",
        "category": "frequency",
    },
    "1": {
        "fr": "un (1)",
        "en": "one (1)",
        "value": 1,
        "category": "number",
    },
    "2": {
        "fr": "deux (2)",
        "en": "two (2)",
        "value": 2,
        "category": "number",
    },
    "3": {
        "fr": "trois (3)",
        "en": "three (3)",
        "value": 3,
        "category": "number",
    },
    "نصف": {
        "fr": "demi",
        "en": "half",
        "value": 0.5,
        "category": "fraction",
    },
    "ماء": {
        "fr": "eau",
        "en": "water",
        "category": "modality",
    },
    "طعام": {
        "fr": "nourriture / repas",
        "en": "food / meal",
        "category": "modality",
    },
    "سكر": {
        "fr": "sucre / diabète",
        "en": "sugar / diabetes",
        "category": "condition",
    },
    "إبرة": {
        "fr": "aiguille / seringue / injection",
        "en": "needle / syringe",
        "category": "instrument",
    },
    "كحول": {
        "fr": "alcool médical / désinfectant",
        "en": "medical alcohol",
        "category": "instrument",
    },
    "طبيب أسنان": {
        "fr": "dentiste",
        "en": "dentist",
        "category": "provider",
    },
    "يساعد": {
        "fr": "aider / besoin d'aide",
        "en": "help",
        "category": "action",
    },
    "من فضلك": {
        "fr": "s'il vous plaît",
        "en": "please",
        "category": "politeness",
    },
    "شكرا": {
        "fr": "merci",
        "en": "thank you",
        "category": "politeness",
    },
    "نعم": {
        "fr": "oui",
        "en": "yes",
        "category": "confirmation",
    },
    "لا": {
        "fr": "non",
        "en": "no",
        "category": "confirmation",
    },
    "حسنا": {
        "fr": "d'accord / bien",
        "en": "okay",
        "category": "confirmation",
    },
    "كم": {
        "fr": "combien",
        "en": "how much",
        "category": "query",
    },
    "مال": {
        "fr": "argent / prix",
        "en": "money / price",
        "category": "query",
    },
}


@dataclass
class SignToken:
    """Represents a single detected sign token with confidence and temporal stamp."""
    label: str
    confidence: float = 1.0
    timestamp: float = 0.0
    french: str = ""
    english: str = ""

    def __post_init__(self):
        meta = SIGN_METADATA.get(self.label, {})
        if not self.french:
            self.french = meta.get("fr", self.label)
        if not self.english:
            self.english = meta.get("en", self.label)


@dataclass
class AssemblyResult:
    """The complete result of translating a sequence of signs."""
    raw_tokens: List[str]
    french_sentence: str
    arabic_sentence: str
    english_sentence: str
    clinical_summary: Dict[str, Any]
    urgency_level: str  # "NORMAL", "ATTENTION", "CRITICAL"


class ClinicalAssembler:
    """Translates temporal sequences of recognized LSA signs into clinical sentences and summaries."""

    def __init__(self, debounce_window_sec: float = 1.5):
        self.debounce_window_sec = debounce_window_sec
        self.tokens: List[SignToken] = []

    def clear(self) -> None:
        """Clear the current sequence buffer."""
        self.tokens.clear()

    def add_token(self, token_or_label: Any, confidence: float = 1.0, timestamp: float = 0.0) -> bool:
        """Add a token with temporal debouncing (ignore consecutive identical signs within debounce window).

        Returns:
            True if the token was added, False if ignored as consecutive duplicate.
        """
        if isinstance(token_or_label, SignToken):
            tok = token_or_label
        else:
            tok = SignToken(label=str(token_or_label), confidence=confidence, timestamp=timestamp)

        # Debouncing: avoid duplicate bursts of the same sign
        if self.tokens:
            last = self.tokens[-1]
            if last.label == tok.label:
                # If timestamps are tracked, check time difference
                if tok.timestamp is not None and last.timestamp is not None and tok.timestamp >= last.timestamp:
                    if (tok.timestamp - last.timestamp) < self.debounce_window_sec:
                        return False
                elif tok.timestamp == 0.0 and last.timestamp == 0.0:
                    return False

        self.tokens.append(tok)
        return True

    def assemble(self) -> AssemblyResult:
        """Analyze accumulated tokens and generate structured clinical record and sentences."""
        if not self.tokens:
            return AssemblyResult(
                raw_tokens=[],
                french_sentence="Aucun signe détecté.",
                arabic_sentence="لم يتم التعرف على أي إشارة.",
                english_sentence="No signs detected.",
                clinical_summary={"status": "empty"},
                urgency_level="NORMAL",
            )

        labels = [t.label for t in self.tokens]
        label_set: Set[str] = set(labels)

        # 1. Slot extraction
        has_please = "من فضلك" in label_set
        has_thank_you = "شكرا" in label_set
        has_help = "يساعد" in label_set
        is_sick = "مريض" in label_set
        has_pain = "ألم" in label_set
        has_fever = "حرارة" in label_set
        has_danger = "خطر" in label_set

        # Anatomy slots
        body_parts = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") == "anatomy"]

        # Temporal slots
        temporals = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") in ("temporal", "frequency")]

        # Posology slots
        numbers = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") in ("number", "fraction")]
        modalities = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") == "modality"]
        conditions = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") == "condition"]
        queries = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") == "query"]
        confirmations = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") == "confirmation"]
        providers = [l for l in labels if SIGN_METADATA.get(l, {}).get("category") == "provider"]

        # Determine urgency
        if has_danger or ("حرارة" in label_set and "الآن" in label_set and "قلب" in label_set):
            urgency = "CRITICAL"
        elif has_fever or (has_pain and "قلب" in label_set):
            urgency = "ATTENTION"
        else:
            urgency = "NORMAL"

        # 2. Compute numeric dosage
        dose_value = 0.0
        for n in numbers:
            val = SIGN_METADATA.get(n, {}).get("value", 0)
            dose_value += val

        # 3. Assemble Clinical Record
        clinical_record: Dict[str, Any] = {
            "urgency": urgency,
            "patient_state": "Malade / Souffrant" if is_sick else "Non spécifié",
            "assistance_demande": has_help,
            "symptomes": [],
            "siege_anatomique": [SIGN_METADATA[bp]["fr"] for bp in body_parts],
            "temporalite": [SIGN_METADATA[t]["fr"] for t in temporals],
            "posologie": {
                "quantite": dose_value if dose_value > 0 else None,
                "frequence": "Chaque jour" if "كل يوم" in label_set else None,
                "avec_eau": "ماء" in label_set,
                "avec_repas": "طعام" in label_set,
            },
            "pathologies_connexes": [SIGN_METADATA[c]["fr"] for c in conditions],
            "specialiste": [SIGN_METADATA[p]["fr"] for p in providers],
        }

        if has_pain:
            clinical_record["symptomes"].append("Douleur")
        if has_fever:
            clinical_record["symptomes"].append("Fièvre")

        # 4. Assemble French Sentence
        fr_parts = []
        if has_danger:
            fr_parts.append("[URGENCE]")
        if has_please:
            fr_parts.append("S'il vous plaît,")
        if has_help:
            fr_parts.append("j'ai besoin d'aide :")

        # State / Symptom / Anatomy construction
        symptom_clauses = []
        if is_sick:
            symptom_clauses.append("je suis malade")

        if has_pain:
            if "رأس" in body_parts:
                symptom_clauses.append("j'ai de violents maux de tête (céphalée)")
            elif "أذن" in body_parts:
                symptom_clauses.append("j'ai mal à l'oreille (otalgie)")
            elif "قلب" in body_parts:
                symptom_clauses.append("j'ai une vive douleur dans la poitrine / au cœur")
            elif "يد" in body_parts:
                symptom_clauses.append("j'ai mal à la main")
            elif "بشرة" in body_parts:
                symptom_clauses.append("j'ai une douleur / irritation de la peau")
            elif "جسم" in body_parts:
                symptom_clauses.append("j'ai des courbatures dans tout le corps")
            else:
                symptom_clauses.append("j'ai des douleurs")

        if has_fever:
            symptom_clauses.append("j'ai de la fièvre")

        if symptom_clauses:
            fr_parts.append(" et ".join(symptom_clauses))

        # Temporals
        if "صباح" in temporals and "الآن" in temporals:
            fr_parts.append("depuis ce matin et encore maintenant.")
        elif "صباح" in temporals:
            fr_parts.append("depuis ce matin.")
        elif "الليل" in temporals:
            fr_parts.append("durant la nuit.")
        elif "الآن" in temporals:
            fr_parts.append("actuellement.")

        # Posology clauses
        poso_clauses = []
        if dose_value > 0:
            if dose_value == 0.5:
                dose_str = "un demi-comprimé"
            elif dose_value == 1.5:
                dose_str = "un comprimé et demi"
            else:
                dose_str = f"{int(dose_value) if dose_value.is_integer() else dose_value} unité(s)"

            freq_str = "chaque jour" if "كل يوم" in label_set else ""
            poso_clauses.append(f"Prendre {dose_str} {freq_str}".strip())

        if "ماء" in modalities:
            poso_clauses.append("avec un grand verre d'eau")
        if "طعام" in modalities:
            poso_clauses.append("au cours du repas")

        if poso_clauses:
            fr_parts.append("Recommandation : " + ", ".join(poso_clauses) + ".")

        if "سكر" in conditions:
            fr_parts.append("(Patient diabétique / surveillance du taux de sucre).")
        if "طبيب أسنان" in providers:
            fr_parts.append("Consultation chez le dentiste recommandée.")

        if "كم" in queries or "مال" in queries:
            fr_parts.append("Combien coûte ce traitement ?")

        if has_thank_you:
            fr_parts.append("Merci beaucoup.")

        if not fr_parts:
            # Fallback to gloss translation
            fr_parts = [SIGN_METADATA.get(l, {}).get("fr", l) for l in labels]

        french_sentence = " ".join(fr_parts).replace(" ,", ",").replace(" .", ".")

        # 5. Assemble Arabic Sentence
        ar_parts = []
        if has_danger:
            ar_parts.append("[حالة طارئة]")
        if has_please:
            ar_parts.append("من فضلك،")
        if has_help:
            ar_parts.append("أحتاج إلى مساعدة:")

        ar_symptom_clauses = []
        if is_sick:
            ar_symptom_clauses.append("أنا مريض")

        if has_pain:
            if "رأس" in body_parts:
                ar_symptom_clauses.append("أعاني من صداع وألم في الرأس")
            elif "أذن" in body_parts:
                ar_symptom_clauses.append("أعاني من ألم في الأذن")
            elif "قلب" in body_parts:
                ar_symptom_clauses.append("أشعر بألم حاد في القلب / الصدر")
            elif "يد" in body_parts:
                ar_symptom_clauses.append("أشعر بألم في اليد")
            elif "بشرة" in body_parts:
                ar_symptom_clauses.append("أعاني من تهيج وألم في الجلد")
            elif "جسم" in body_parts:
                ar_symptom_clauses.append("أشعر بآلام عامة في الجسم")
            else:
                ar_symptom_clauses.append("أعاني من ألم")

        if has_fever:
            ar_symptom_clauses.append("لدي ارتفاع في درجة الحرارة")

        if ar_symptom_clauses:
            ar_parts.append(" و ".join(ar_symptom_clauses))

        if "صباح" in temporals:
            ar_parts.append("منذ الصباح.")
        elif "الليل" in temporals:
            ar_parts.append("خلال الليل.")
        elif "الآن" in temporals:
            ar_parts.append("في الوقت الحالي.")

        if dose_value > 0:
            if dose_value == 0.5:
                dose_ar = "نصف قرص"
            elif dose_value == 1.5:
                dose_ar = "قرص ونصف"
            elif dose_value == 1:
                dose_ar = "قرص واحد"
            elif dose_value == 2:
                dose_ar = "قرصان"
            else:
                dose_ar = f"{int(dose_value)} أقراص"

            freq_ar = "كل يوم" if "كل يوم" in label_set else ""
            ar_parts.append(f"الجرعة: تناول {dose_ar} {freq_ar}".strip())

        if "ماء" in modalities:
            ar_parts.append("مع الماء")
        if "طعام" in modalities:
            ar_parts.append("أثناء الطعام")

        if "سكر" in conditions:
            ar_parts.append("(مريض سكري).")
        if "طبيب أسنان" in providers:
            ar_parts.append("يرجى مراجعة طبيب الأسنان.")

        if "كم" in queries or "مال" in queries:
            ar_parts.append("كم سعر الدواء؟")

        if has_thank_you:
            ar_parts.append("شكراً جزيلاً.")

        if not ar_parts:
            ar_parts = labels

        arabic_sentence = " ".join(ar_parts)

        # 6. Assemble English Sentence
        en_parts = []
        if has_danger:
            en_parts.append("[EMERGENCY]")
        if has_please:
            en_parts.append("Please,")
        if has_help:
            en_parts.append("I need assistance:")

        en_symptoms = []
        if is_sick:
            en_symptoms.append("I am feeling unwell")
        if has_pain:
            if "رأس" in body_parts:
                en_symptoms.append("I have a headache")
            elif "أذن" in body_parts:
                en_symptoms.append("I have ear pain")
            elif "قلب" in body_parts:
                en_symptoms.append("I have chest/heart pain")
            else:
                en_symptoms.append("I am experiencing pain")
        if has_fever:
            en_symptoms.append("I have a fever")

        if en_symptoms:
            en_parts.append(" and ".join(en_symptoms))

        if "صباح" in temporals:
            en_parts.append("since this morning.")
        elif "الليل" in temporals:
            en_parts.append("during the night.")
        elif "الآن" in temporals:
            en_parts.append("right now.")

        if dose_value > 0:
            en_parts.append(f"Dosage: {dose_value} unit(s) {'daily' if 'كل يوم' in label_set else ''}.")
        if "ماء" in modalities:
            en_parts.append("Take with water.")
        if "طعام" in modalities:
            en_parts.append("Take with meals.")

        if has_thank_you:
            en_parts.append("Thank you.")

        english_sentence = " ".join(en_parts) if en_parts else " ".join([SIGN_METADATA.get(l, {}).get("en", l) for l in labels])

        return AssemblyResult(
            raw_tokens=labels,
            french_sentence=french_sentence,
            arabic_sentence=arabic_sentence,
            english_sentence=english_sentence,
            clinical_summary=clinical_record,
            urgency_level=urgency,
        )
