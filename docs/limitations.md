# Algerian Sign Language (LSA) Model Limitations & Clinical Boundaries

## 1. Domain Shift: Synthetic Avatars vs. Real Human Signers
The training data (VDzSL) consists exclusively of 3D synthetic avatars animated via CWA Signing Avatars software from SiGML definitions.
### Critical Limitations:
- **Visual Texture & Geometry**: Avatars lack natural skin pores, realistic muscle contractions, finger elasticity, and complex skin tones.
- **Lighting & Background Invariance**: The dataset was rendered under uniform, artificial studio lighting with clean neutral backgrounds. Real pharmacy environments exhibit varied illumination, fluorescent flicker, and cluttered background shelves.
- **Motion Dynamics**: Synthetic avatars move along rigid motion interpolation curves, whereas human signers demonstrate natural speed variations, dialectal co-articulation, and non-manual facial markers.
- **Detector Failure on Synthetic Hands**: Empirical evaluation revealed that MediaPipe Hand detection dropped to 27.6%–36.1% on `signer2` due to synthetic shading artifacts.

---

## 2. Structural Limitation: Isolated Signs vs. Continuous Translation
- **Isolated Classification**: The model processes individual, pre-segmented video clips corresponding to single isolated signs (mean duration 0.80 seconds).
- **Continuous Signing is Unsupported**: VDzSL contains **no sentence-level recordings**, no sign boundary annotations, and no translation alignments.
- **No Grammatical Translation**: The model cannot infer Algerian Sign Language grammar, topicalization, directional verb morphology, or facial syntax. Concatenating isolated sign predictions does **not** constitute sign language translation.

---

## 3. Lexical Coverage Gaps in Healthcare Domain
While 32 useful signs have been verified (symptoms like `ألم`, `حرارة`, anatomy like `رأس`, `يد`, `قلب`, `أذن`, `بشرة`, and dosage numbers `1`, `2`, `3`, `نصف`), the dataset lacks fundamental pharmaceutical words:
- `دواء` (Medicine / Drug) is **absent**.
- `صيدلية` (Pharmacy) is **absent**.
- `مستشفى` (Hospital / Clinic) is **absent**.
- `طبيب` (General Physician / Doctor) is **absent** (only `طبيب أسنان` Dentist is present).
- `حساسية` (Allergy) is **absent**.
- `جرعة` / `حبة` (Dosage / Tablet) is **absent**.

> [!WARNING]
> **Strict Prohibition Against Semantic Hallucination**:
> The model must **never** substitute an absent concept (e.g. predicting `طبيب أسنان` when a user signs `طبيب`, or guessing `دواء` from an uncertain sign).

---

## 4. Medical Safety Mandate
This software is an AI research prototype for **communication support**, NOT a medical device.
- **No Diagnosis**: It must never be used to diagnose a symptom.
- **No Prescribing**: It must never confirm or recommend drug dosages.
- **Mandatory Written Confirmation**: All safety-critical instructions (dosage counts, allergy status, contraindications) require explicit written/visual confirmation between the pharmacist and the deaf patient.
- **Uncertainty Rejection**: Predictions with confidence below the calibrated threshold must be rejected with an explicit "Uncertain / Recheck" prompt.
