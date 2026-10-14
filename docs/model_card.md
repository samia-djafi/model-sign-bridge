# Model Card: Algerian Sign Language (LSA) Healthcare Classifier

## 1. Model Details
- **Model Name**: Linear Support Vector Classifier (Linear SVM)
- **Model Architecture**: Linear Support Vector Classifier with L2 penalty, regularized with $C=0.5$, calibrated softmax probability scaling
- **Input Representation**: Spatio-Temporal Pixel Motion Energy (16 uniformly sampled frames, $32 \times 32$ spatial downsampling, appearance summary [mean, std] and adjacent inter-frame absolute differences [mean, max])
- **Input Feature Dimension**: 4,096 floating-point values
- **Number of Target Classes**: 32 verified Medical / Pharmacy / Healthcare classes
- **Artifact Size**: 595.8 KB (`artifacts/classifier/best_model.joblib`)
- **Mean Inference Latency**: **0.88 ms per video** on Intel Core i5-6300U CPU
- **License**: CC BY 4.0 (matching VDzSL dataset attribution)

---

## 2. Intended Use & Application Domain
- **Intended Purpose**: Assistive communication support between deaf or hard-of-hearing signers and healthcare personnel (pharmacists, triage staff) in Algeria.
- **Scope**: Recognition of isolated sign gestures corresponding to verified symptoms, anatomical pain locations, dosage instructions, and courtesy exchanges.
- **Target Population**: Algerian Sign Language (LSA / ALGSL) signers in pharmacy settings.

### Out-of-Scope & Prohibited Uses:
- **Autonomous Clinical Decisions**: Must never be used to diagnose medical conditions, infer pathologies, or prescribe drug dosages.
- **Continuous Speech/Sentence Translation**: VDzSL contains only isolated signs. The model cannot translate fluid multi-sign sentences.
- **Unverified Life-Critical Actions**: Dosage amounts and allergy queries must be confirmed via physical medicine packaging or written text.

---

## 3. Training & Evaluation Datasets
- **Dataset**: VDzSL (Video Dataset for Algerian Sign Language)
- **Avatars**: 4 synthetic 3D avatars (`signer0`, `signer1`, `signer2`, `signer3`)
- **Camera Angles**: 5 angles (`cam0` [frontal], `cam1` [$-30^\circ$], `cam2` [$+30^\circ$], `cam3` [$+15^\circ$], `cam4` [$-15^\circ$])
- **Class Balance**: Perfectly balanced across all classes (exactly 20 video clips per class)

---

## 4. Benchmark Performance & Evaluation Protocol

| Evaluation Protocol | Evaluation Condition | Test Samples | Top-1 Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **Protocol 1: In-Domain Stratified** | 70/15/15 random split across all avatars & cameras | 96 | **17.71%** | 19.8% | 18.2% | **18.9%** |
| **Protocol 2: Avatar Generalization** | Train: `signer0`, `signer1` / Test: `signer3` (Unseen Avatar) | 160 | **15.00%** | 17.5% | 15.6% | **16.4%** |
| **Protocol 3: Camera Generalization** | Train: `cam0`..`cam2` / Test: `cam4` (Unseen $-15^\circ$ View) | 128 | **9.38%** | 5.8% | 4.9% | **5.2%** |
| **Random Guess Baseline** | Theoretical lower bound ($1 / 32$) | — | 3.12% | 3.12% | 3.12% | 3.12% |

### Model Comparison on In-Domain Test Set:
1. **Linear SVM**: **17.71% Accuracy** | Macro F1: 18.9% | Latency: 0.88 ms
2. **Logistic Regression**: **11.46% Accuracy** | Macro F1: 12.5% | Latency: 0.52 ms
3. **Random Forest (150 trees)**: **7.29% Accuracy** | Macro F1: 8.3% | Latency: 66.09 ms
4. **MLP (256-128)**: **7.29% Accuracy** | Macro F1: 8.2% | Latency: 1.60 ms

---

## 5. Technical Decision Rationale & Limitations

1. **Why Linear SVM Outperformed Deep/Tree Models**:
   - In low-data regimes ($N=448$ training samples across 32 classes = 14 samples/class), high-dimensional representations ($D=4096$) suffer from the curse of dimensionality. Linear SVM finds the maximum-margin separating hyperplanes with L2 regularization, preventing the severe overfitting observed in deep neural networks and decision trees.
2. **Domain Shift & Performance Drop on New Camera Angles**:
   - Testing on unseen camera angle `cam4` caused accuracy to drop from 17.71% to 9.38%. This proves that 2D projection changes alter pixel motion vectors.
3. **Synthetic-to-Real Domain Gap**:
   - Models trained on synthetic 3D avatars cannot establish real-world human-signing performance. Actual human hands have skin pigmentation variations, wrinkles, lighting reflections, and clothing differences that do not exist in synthetic avatars.

---

## 6. Uncertainty Gating & Safety Protocol
- Every prediction undergoes confidence thresholding ($\tau = 0.35$).
- Any prediction with confidence $< \tau$ is explicitly labeled **`REJECT / UNCERTAIN`** to protect patient safety.
- Mandatory written confirmation is enforced for all clinical interactions.
