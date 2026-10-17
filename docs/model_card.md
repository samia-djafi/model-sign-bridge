# Model Card: Algerian Sign Language (LSA) Healthcare Classifier

## 1. Model Details
- **Model Name**: Linear Support Vector Classifier (Linear SVM) on Landmark Features
- **Model Architecture**: Linear Support Vector Classifier with L2 penalty, regularized with $C=0.1$, calibrated softmax probability scaling
- **Input Representation**: MediaPipe 3D Hand and Pose Landmarks with scale invariance, anatomical finger geometry, palm normal orientation, and temporal trajectory dynamics:
  - 16 uniformly sampled frames
  - Upper body pose (Nose 0, Shoulders 11/12, Elbows 13/14, Wrists 15/16) centered on shoulders midpoint, scaled by shoulder distance
  - Left & Right hands: 21 landmarks centered on wrist and scaled by hand size ($||MCP9 - wrist||$)
  - Hand presence flags, relative wrist locations, palm normal vector, fingertip extension distances, pinch distance
  - Temporal dynamics: mean, std, mean frame-to-frame velocity, wrist trajectory net displacement, cumulative path length, and coordinate range
- **Input Feature Dimension**: 3,453 floating-point values
- **Number of Target Classes**: 32 verified Medical / Pharmacy / Healthcare classes
- **Artifact Size**: 833.3 KB (`artifacts/classifier/best_model.joblib`)
- **Mean Inference Latency**: **1.15 ms per sample** on CPU
- **License**: CC BY 4.0 (matching VDzSL dataset attribution)

---

## 2. Intended Use & Application Domain
- **Intended Purpose**: Assistive communication support between deaf or hard-of-hearing signers and healthcare personnel (pharmacists, triage staff) in Algeria.
- **Scope**: Recognition of isolated sign gestures corresponding to verified symptoms, anatomical pain locations, dosage instructions, and courtesy exchanges.
- **Target Population**: Algerian Sign Language (LSA / ALGSL) signers in pharmacy settings.

### Out-of-Scope & Prohibited Uses:
- **Autonomous Clinical Decisions**: Must never be used to diagnose medical conditions, infer pathologies, or prescribe drug dosages.
- **Continuous Speech/Sentence Translation**: VDzSL contains only isolated signs. The model cannot translate fluid multi-sign sentences without clinical slot-filling supervision.
- **Unverified Life-Critical Actions**: Dosage amounts and allergy queries must be confirmed via physical medicine packaging or written text.

---

## 3. Training & Evaluation Datasets
- **Dataset**: VDzSL (Video Dataset for Algerian Sign Language)
- **Avatars**: 4 synthetic 3D avatars (`signer0`, `signer1`, `signer2`, `signer3`)
- **Camera Angles**: 5 angles (`cam0` [frontal], `cam1` [$-30^\circ$], `cam2` [$+30^\circ$], `cam3` [$+15^\circ$], `cam4` [$-15^\circ$])
- **Class Balance**: Perfectly balanced across all classes (exactly 20 video clips per class, 640 total clips)

---

## 4. Benchmark Performance & Evaluation Protocol

### Landmark Model vs. Previous Pixel-Motion Baseline:

| Evaluation Protocol | Evaluation Condition | Test Samples | Pixel Baseline Top-1 | **New Landmark Top-1** | Baseline Macro F1 | **New Landmark Macro F1** | Absolute Improvement |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **Protocol 1: In-Domain Stratified** | 70/15/15 random split | 96 | 17.71% | **67.71%** | 18.89% | **66.16%** | **+50.00% (3.8x)** |
| **Protocol 2: Avatar Generalization** | Train: `signer0,1` / Test: `signer3` (Unseen) | 160 | 15.00% | **65.62%** | 16.43% | **64.57%** | **+50.62% (4.4x)** |
| **Protocol 3: Camera Generalization** | Train: `cam0..2` / Test: `cam4` (Unseen $-15^\circ$) | 128 | 9.38% | **58.59%** | 5.21% | **57.77%** | **+49.21% (6.2x)** |
| **Random Guess Baseline** | Theoretical lower bound ($1 / 32$) | — | 3.12% | 3.12% | 3.12% | 3.12% | — |

### Model Comparison on In-Domain Landmark Test Set:
1. **Linear SVM ($C=0.1$)**: **67.71% Accuracy** | **Macro F1: 66.16%** | Latency: **1.15 ms**
2. **Random Forest (150 trees)**: **64.58% Accuracy** | Macro F1: 64.33% | Latency: 65.32 ms
3. **Logistic Regression ($C=1.0$)**: **51.04% Accuracy** | Macro F1: 47.12% | Latency: 0.64 ms
4. **MLP (128-64)**: **34.38% Accuracy** | Macro F1: 34.02% | Latency: 1.06 ms

---

## 5. Technical Analysis & Engineering Rationale

1. **Why Landmark Normalization Dramatically Improved Generalization**:
   - The pixel baseline overfitted to synthetic clothing, skin textures, and background lighting. By centering coordinates on the torso and wrist, normalizing by shoulder distance, and computing explicit palm normals and finger extension geometries, the landmark pipeline achieved **58.59% accuracy on held-out camera angles** (up from 9.38%) and **65.62% on unseen avatars** (up from 15.00%).
2. **Linear SVM Superiority in Low-Data Regimes**:
   - With 448 training samples across 32 classes (~14 samples per class), high-capacity neural networks (MLP) overfit. Linear SVM finds the maximum-margin decision boundary with L2 regularization, delivering the best balance of accuracy, generalization, and sub-2ms inference latency on CPU.
3. **Clinical Guardrails**:
   - Predictions below the uncertainty threshold ($\tau = 0.35$) trigger an explicit `REJECT / UNCERTAIN` flag.
   - Low-confidence predictions are blocked from entering the `ClinicalAssembler` sentence stream.
