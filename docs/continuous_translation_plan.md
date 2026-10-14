# Continuous Algerian Sign Language (LSA) Translation: Research Plan & Feasibility Assessment

## 1. Ground Truth Feasibility Determination for VDzSL
**Definitive Finding**:
The **VDzSL dataset does NOT support continuous sign language translation**.
### Empirical Evidence from Dataset Audit:
1. **Isolated Nature**: All 8,300 video files correspond to single, isolated lexical signs (mean duration: 0.80 seconds, 37.5 frames).
2. **Absence of Continuous Signing**: There are zero recordings of multi-sign sequences, conversations, or fluid continuous sentences.
3. **Absence of Temporal Annotations**: There are no frame-level sign boundary markers ($t_{start}, t_{end}$) or co-articulation transition labels.
4. **Absence of Sentence Translations**: There are no paired spoken language (Arabic / French) parallel sentences.

> [!IMPORTANT]
> **Scientific Integrity Requirement**:
> An isolated sign classifier must **never** be described as a continuous sign language translator. Concatenating predicted isolated sign glosses (e.g. `[ماء] + [صباح] + [1]`) does **not** constitute natural sign language translation, as Algerian Sign Language exhibits its own spatial grammar, classifier predicates, topic-comment word order, and facial non-manual syntax.

---

## 2. Requirements for a Genuine Continuous LSA Translation System

To develop a defensible continuous translation system in a future phase, a dedicated data collection protocol is required:

### 2.1 Required Data Modalities
1. **Continuous Video Recordings**: Multi-turn dialogue between deaf signers in realistic healthcare / pharmacy scenarios.
2. **Two-Level Annotations**:
   - **Gloss-Level Annotations**: Temporal start and end timestamps for each constituent sign gesture, including transitional epenthesis movements.
   - **Sentence-Level Parallel Translations**: Grammatically natural translation into Modern Standard Arabic (MSA), Algerian Darja, and French.
3. **Non-Manual Signals**: Dedicated annotation of facial expressions (eyebrow raise for questions, head nods/shakes for negation, mouthings/mouth gestures).

### 2.2 Ethics, Consent, and Community Partnership
- **Native Deaf Community Collaboration**: Partnerships with Algerian deaf associations and certified LSA interpreters.
- **Informed Consent**: Explicit multi-tiered consent covering research data use, facial display, and open vs. restricted research distribution.
- **Privacy Protection**: Feature-level distribution (e.g., anonymized 3D coordinate trajectories) to prevent biometric facial identification when required by participants.

---

## 3. Recommended Technical Architecture for Continuous Translation

When continuous paired data becomes available, the recommended architecture decomposes the problem into two decoupled stages:

```
[Continuous Video Stream (T frames)]
                │
                ▼
┌──────────────────────────────────────────────┐
│ Stage 1: Continuous Sign Recognition (CSLR)  │
│  - Spatial Feature Extractor (CNN / ViT)     │
│  - Temporal Sequence Model (Bi-LSTM / 1D-TCN)│
│  - Connectionist Temporal Classification     │
│    (CTC Loss with Gloss Alignment)           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼ Predicted Gloss Sequence (e.g., GLOSS_PAIN HEAD MORNING)
┌──────────────────────────────────────────────┐
│ Stage 2: Sign-to-Text Translation (SLT)      │
│  - Sequence-to-Sequence Transformer / LLM    │
│  - Adapts LSA Grammar -> Target Natural Text │
│  - Cross-Entropy Translation Loss            │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
   "I have had a severe headache since this morning." (French / Arabic text)
```

### Why a Decomposed Pipeline?
1. **Low-Resource Data Efficiency**: Continuous sign language datasets are typically small ($10^3$ to $10^4$ sentences). End-to-end video-to-text models overfit severely. Decoupling recognition (CTC) from translation (pretrained linguistic models) allows leveraging pretrained text language models.
2. **Interpretability & Error Localization**: Pharmacists and interpreters can inspect whether an error originated in visual sign detection or language generation.

---

## 4. Interim Phase Strategy
Until continuous LSA clinical data is acquired:
- Maintain focus on the **isolated-sign communication support tool** with verified confidence and uncertainty rejection.
- Implement live temporal debouncing and motion windowing (as built in `scripts/webcam_recognition.py`) to handle live frame streaming gracefully without making unsubstantiated translation claims.
