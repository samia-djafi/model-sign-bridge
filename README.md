# Algerian Sign Language (LSA/ALGSL) AI Research Project

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Intel_i5_CPU_Only-orange.svg)](#hardware-and-environment)

A scientifically defensible, reproducible Machine Learning and Computer Vision system for **Algerian Sign Language (LSA / لغة الإشارة الجزائرية)** recognition, with an initial domain focus on **Medical - Pharmacy and Basic Healthcare Communication**.

---

## 1. Project Scope & Architecture Boundary

- **Independent AI Research Repository**: This is a standalone ML research project. It has zero dependencies on any external frontend, web application, or third-party cloud API.
- **CPU-First Pipeline**: Optimized for local Intel i5 hardware (2 cores, 4 threads, 8 GB RAM) without assuming or requiring a dedicated NVIDIA GPU.
- **Memory-Safe Streaming**: Video decoding is performed sequentially and in small batches; reusable features are extracted once and cached to disk.

---

## 2. Dataset: VDzSL (Audited)

- **Official Dataset**: [VDzSL on Zenodo](https://zenodo.org/records/18209839) (DOI: `10.5281/zenodo.18209839`)
- **Research Article**: [Data in Brief / PMC12925472](https://pmc.ncbi.nlm.nih.gov/articles/PMC12925472/)
- **Total Videos**: Exactly 8,300 files (100% verified, 0 corrupted files).
- **Structure**: 4 avatars (`signer0`..`signer3`) $\times$ 5 camera angles (`cam0`..`cam4`) $\times$ 415 distinct signs.
- **Linguistic Nature**: **Isolated signs only**. The dataset contains **no continuous signing** and **no sentence-level annotations**.

---

## 3. Directory Layout

```
bridge_sign_model/
├── README.md                      # Project documentation and execution instructions
├── requirements.txt               # CPU-friendly dependencies
├── pyproject.toml                 # Editable package specification
├── pytest.ini                     # Automated testing configuration
├── configs/                       # Hyperparameter and experiment configurations
├── data/
│   ├── VDzSL/                     # Raw video dataset (4 signers x 5 cameras x 415 signs)
│   ├── manifests/                 # Full inventory, class frequencies, and split manifests
│   └── processed/                 # Cached feature arrays (.npz)
├── docs/
│   ├── dataset_audit.md           # Dataset verification, balance, and encoding audit
│   ├── vocabulary_audit.md        # Medical/pharmacy vocabulary analysis
│   ├── model_card.md              # Model architecture, performance, and conditions
│   ├── limitations.md             # Domain shift, synthetic gap, and clinical boundaries
│   └── continuous_translation_plan.md # Research roadmap for continuous signing
├── scripts/
│   ├── audit_dataset.py           # Full 8,300-video audit and manifest generation
│   ├── audit_vocabulary.py        # Healthcare vocabulary audit and gap analysis
│   ├── test_landmarks.py          # Empirical MediaPipe benchmark across avatars and angles
│   ├── build_manifest.py          # Leakage-aware split generation (in-domain, avatar, camera)
│   ├── preprocess_videos.py       # Spatio-temporal feature extraction and caching
│   ├── train_classifier.py        # Multi-model benchmarking across evaluation protocols
│   ├── predict_video.py           # Command-line video inference with uncertainty rejection
│   └── webcam_recognition.py      # Real-time webcam experiment with motion debouncing
├── src/
│   ├── data/                      # LabelParser, dataset inventory, and split generators
│   ├── preprocessing/             # FrameSampler (uniform sampling, padding, resizing)
│   ├── features/                  # PixelMotionFeatureExtractor & LandmarkFeatureExtractor
│   ├── models/                    # SignClassifier (SVM, Logistic Regression, Random Forest, MLP)
│   ├── evaluation/                # Classification metrics, confusion matrix, and reports
│   ├── inference/                 # VideoSignPredictor with calibrated uncertainty gating
│   └── webcam/                    # LiveSignRecognizer (rolling buffer and motion activity)
├── artifacts/
│   ├── classifier/                # Trained model weights and config (best_model.joblib)
│   ├── label_mappings/            # Canonical class JSON mappings
│   ├── models/                    # MediaPipe task bundles
│   └── reports/                   # Model comparison tables and confusion matrices
└── tests/                         # Pytest test suite (15 automated unit tests)
```

---

## 4. Setup & Installation

### Step 1: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
pip install -e . --no-deps
```

### Step 3: Run the Automated Unit Test Suite
```powershell
pytest
```
All 29 tests verify Unicode label parsing, whitespace normalization, uniform frame sampling, leakage-aware data splitting, landmark feature extraction, hand skeleton visualization, and clinical session management.

---

## 5. Execution Pipeline (Checkpoints 1 through 8)

### 1. Audit the Dataset
```powershell
python scripts/audit_dataset.py --workers 4
```
Generates `data/manifests/dataset_inventory.csv` and `docs/dataset_audit.md`.

### 2. Audit Medical & Pharmacy Vocabulary
```powershell
python scripts/audit_vocabulary.py
```
Filters the 32 verified healthcare MVP classes and documents lexical gaps in `docs/vocabulary_audit.md`.

### 3. Build Leakage-Aware Data Splits
```powershell
python scripts/build_manifest.py
```
Creates:
- `split_mvp_stratified.csv`: 70% train, 15% val, 15% test.
- `split_mvp_avatar_heldout.csv`: Train on `signer0`, `signer1`; test on `signer3` (unseen avatar).
- `split_mvp_camera_heldout.csv`: Train on `cam0`..`cam2`; test on `cam4` (unseen camera angle).

### 4. Extract and Cache Features
```powershell
# Extract landmark features with multiprocessing (recommended, 3453-dim):
python scripts/preprocess_videos.py --mode landmark --workers 4

# Or extract legacy pixel motion features (4096-dim):
python scripts/preprocess_videos.py --mode pixel
```
Extracts scale-normalized MediaPipe hand & pose landmarks with temporal trajectory dynamics into `data/processed/features_mvp_landmark.npz` (5.38 MB).

### 5. Train & Benchmark Classifiers
```powershell
python scripts/train_classifier.py
```
Trains and benchmarks Linear SVM, Logistic Regression, Random Forest, and MLP across in-domain and generalization protocols, achieving **67.71% in-domain accuracy**, **65.62% on unseen avatars**, and **58.59% on unseen camera angles**, saving `artifacts/classifier/best_model.joblib`.

### 6. Predict on a Video File
```powershell
python scripts/predict_video.py --video "data/VDzSL/signer0/cam0/ألم.mp4"
```
Outputs the predicted Arabic sign, verified English/French translation, confidence score, and uncertainty status.

### 7. Run the Live Webcam Experiment
```powershell
# Using physical webcam (Device 0):
python scripts/webcam_recognition.py --camera-id 0

# Or simulating live input using a video clip:
python scripts/webcam_recognition.py --video-simulation "data/VDzSL/signer0/cam0/ألم.mp4"
```
Features real-time 21 hand landmarks and anatomical skeleton overlay, confidence gating, debounced clinical slot-filling (`ClinicalAssembler`), and interactive controls (`[P]` Pause/Resume, `[C]` Clear, `[F]` Finalize Fiche Comptoir, `[Q]` Exit).

---

## 6. Medical Safety Protocol

> [!CAUTION]
> **Clinical Boundaries**:
> 1. This software is an **investigational communication-support prototype**, NOT a diagnostic medical device.
> 2. The model must **never** infer a medical disease, diagnose a condition, or recommend medication dosage.
> 3. All safety-critical instructions (dosage counts, allergies, administration schedules) require **mandatory written/visual confirmation** between the pharmacist and the patient.
> 4. Any prediction below the calibrated confidence threshold triggers an explicit **"UNCERTAIN / RECHECK"** state.
