# BridgeSign — Algerian Sign Language (LSA) Data Collection Protocol

## 1. Overview and Mission
This document establishes the official data-collection and annotation protocol for expanding **BridgeSign — Every Hand Has a Voice**, an Algerian Sign Language (*Langue des Signes Algérienne* / لغة الإشارة الجزائرية - LSA) accessibility system tailored for medical, pharmaceutical, and emergency communication.

The protocol ensures that all collected sign video recordings maintain high linguistic validity, anatomical clarity for computer vision models, ethical participant protections, and consistent annotation standards.

---

## 2. Ethical Standards and Participant Protections

### 2.1 Informed Consent and IRB Approval
- All data collection sessions must operate under approved institutional ethical protocols with explicit written and sign-recorded informed consent.
- Consent documents must be provided in both bilingual written format (Arabic, French) and accessible video-recorded LSA explanations by a deaf advocate.
- Participants retain the right to withdraw from the dataset at any stage prior to model training, resulting in immediate purging of raw footage.

### 2.2 Privacy, Anonymity, and Facial Data Handling
- **MediaPipe Landmark Separation**: Computer vision feature extraction uses coordinate landmarks (21 hand joints + upper body pose). Landmark coordinates cannot be reversed to reconstruct raw photographic likeness.
- **Visual De-identification**: For publicly shareable validation sets, face-blurring or animated avatar transfer (similar to the synthetic VDzSL rendering methodology) must be applied unless explicit photographic consent was granted.
- **Secure Data Storage**: Raw video recordings must be stored on encrypted storage drives (AES-256) with role-based access control limited to authorized research engineers.

---

## 3. Participant Recruitment and Demographics

### 3.1 Signer Qualifications
To prevent non-standard or invented signs from contaminating the clinical lexicon:
- **Primary Signers**: Deaf individuals who use Algerian Sign Language as their primary native or habitual language of daily communication.
- **Linguistic Advisors**: Certified Algerian Sign Language interpreters and representatives from Algerian deaf associations (e.g., *Fédération Nationale des Sourds d'Algérie* - FNSA).
- **Medical Validation**: Clinical concepts must be reviewed by deaf healthcare workers or interpreters experienced in Algerian hospital and dispensary consultations.

### 3.2 Diversity and Demographic Stratification
- **Geographic Representation**: Include signers from major Algerian regions (Northern/Central Algiers, Western Oran, Eastern Constantine/Annaba, Southern Ghardaïa/Ouargla) to document and accommodate regional lexical variations.
- **Physical Characteristics**: Stratify signers across hand dimensions, skin tones, ages (18–70), and signing speeds (slow deliberate vs. rapid natural).

---

## 4. Recording Studio Setup & Hardware Specifications

### 4.1 Camera Geometry & Multi-Angle Configuration
Following and extending the VDzSL multi-camera benchmark:
- **Primary Camera (Frontal, 0°)**: Positioned at chest level, 1.2 to 1.5 meters from the signer, capturing the signing space from the waist to 30 cm above the head.
- **Secondary Angles**:
  - Angle 1: +30° right oblique.
  - Angle 2: -30° left oblique.
  - Angle 3: +15° elevated (simulating healthcare desk/countertop perspective).
  - Angle 4: -15° downward or side profile (for depth ambiguities such as fingerspelling hand extensions).

### 4.2 Environmental & Visual Requirements
- **Resolution and Frame Rate**: Minimum 1080p (1920×1080) at 60 fps (or stable 30 fps) with fixed shutter speed (1/120s or faster) to prevent motion blur during rapid finger movements.
- **Lighting**: Three-point diffused LED lighting (key, fill, back) eliminating harsh shadows on the hands and chest.
- **Attire and Background**: Signers wear solid, non-patterned, dark clothing contrasting with skin tone. Clean neutral backdrop (chroma grey or blue) free of visual clutter or reflective surfaces.

---

## 5. Annotation Methodology and Quality Control

### 5.1 Temporal Boundary Segmentation
Every continuous sign video must be segmented into three distinct temporal phases:
1. **Preparation Phase ($t_{\text{prep}}$)**: Movement of hand(s) from rest or neutral position toward the signing space.
2. **Stroke / Apex Phase ($t_{\text{start}} \dots t_{\text{end}}$)**: The canonical lexical execution of the sign, including handshape, orientation, location, and movement trajectory. **This phase constitutes the primary feature extraction window.**
3. **Retraction Phase ($t_{\text{retract}}$)**: Return of hand(s) to neutral rest or transition to the subsequent sign.

### 5.2 Linguistic Metadata Tagging
Each clip must be annotated with:
- **Concept ID**: Official catalog ID (e.g., `MED-001`, `EMG-005`, `ALPHA-01`).
- **Handedness**: Dominant right, dominant left, or ambidextrous two-handed.
- **Dialect / Regional Variant Tag**: `DZ-ALG` (Algiers), `DZ-ORN` (Oran), `DZ-CST` (Constantine), `DZ-STD` (Standard).
- **Execution Speed**: Normal, slow, fast.

### 5.3 Inter-Annotator Agreement (IAA)
- A minimum 15% random sample of recordings must be independently annotated by two certified sign linguistics raters.
- Cohen’s Kappa ($\kappa$) for categorical sign classification must exceed **0.90**.
- Temporal intersection-over-union (IoU) for stroke boundaries must exceed **0.85**. Discrepancies are resolved by a consensus committee.

---

## 6. Prioritized Data Collection Backlog

| Phase | Category | Target Scope | Existing VDzSL Count | Needed Collections | Minimum Target Clips |
|---|---|---|---|---|---|
| **Phase 1** | **Algerian Alphabet (Manual Fingerspelling)** | 31 letters (`ALPHA-01`..`31`) | **0 clips** | **31 letters** | **3,100 clips** (10 signers × 10 reps) |
| **Phase 2** | **Emergency Vocabulary** | 30 urgent concepts (`EMG-001`..`030`) | **10 concepts** | **20 concepts** | **2,000 clips** (10 signers × 10 reps) |
| **Phase 3** | **Missing Medical & Pharmacy Concepts** | 100 concepts (`MED-001`..`100`) | **45 concepts** | **55 concepts** | **5,500 clips** (10 signers × 10 reps) |
| **Phase 4** | **General Healthcare & Dialectal Variants** | 20 concepts (`GEN-001`..`020`) | **12 concepts** | **8 concepts** + variants | **1,500 clips** (10 signers × 10 reps) |

### Milestone Acceptance Criteria
- Models must not transition from `data_missing` to `data_available` until raw clips satisfy the 1080p/60fps criteria with signed consent forms on file.
- Models must not transition to `training_complete` until cross-signer held-out F1 score exceeds 0.70 across the collected category.
