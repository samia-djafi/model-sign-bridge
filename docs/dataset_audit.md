# VDzSL Dataset Audit Report

## 1. Executive Summary
- **Dataset Name**: VDzSL (Video Dataset for Algerian Sign Language)
- **Source**: Zenodo DOI `10.5281/zenodo.18209839` / PMC `PMC12925472`
- **Total Video Files Discovered**: 8,300
- **Successfully Decoded & Verified**: 8,300 (100.0%)
- **Corrupted or Unreadable Files**: 0
- **Unique Canonical Classes**: 415 (21 numeric digit signs, 394 Arabic text signs)
- **Class Balance**: Perfectly balanced (exactly 20 videos per class)
- **Video Resolution**: 498x498: 8300 (100.0%)
- **Average Duration**: 0.80 seconds (Range: 0.12s – 2.87s)
- **Average Frame Count**: 37.5 frames (Median: 35.0, Min: 14, Max: 90, Std: 12.9)

---

## 2. Linguistic and Structural Nature of the Data
Based on the code inspection, file naming, and research paper:
1. **Isolated Signs**: **YES**. Each video recording represents a single, isolated sign gesture (e.g., `أخ` [brother], `طبيب أسنان` [dentist], `1`..`10` [digits]).
2. **Continuous Signing Sequences**: **NO**. The dataset contains **no continuous signing sequences**, no temporal sign-to-sign transitions, and no sentence-level dialogue recordings.
3. **Sentence Translations**: **NO**. There are no sentence boundary annotations or spoken sentence translations in VDzSL.
4. **Synthetic Avatar Domain**: All 8,300 videos were generated using **3D synthetic avatars** rendered from SiGML (Signing Gesture Markup Language) files via CWA Signing Avatars software. **No human signers are present in this dataset.**

---

## 3. Demographics and Camera Setup
### 3.1 Avatars / Signers
| Signer Identifier | Video Count | Percentage | Type |
| :--- | :--- | :--- | :--- |
| `signer0` | 2,075 | 25.0% | Synthetic 3D Avatar |
| `signer1` | 2,075 | 25.0% | Synthetic 3D Avatar |
| `signer2` | 2,075 | 25.0% | Synthetic 3D Avatar |
| `signer3` | 2,075 | 25.0% | Synthetic 3D Avatar |

### 3.2 Camera Angles
| Camera Identifier | Video Count | Percentage | Polar Coordinates | Description |
| :--- | :--- | :--- | :--- | :--- |
| `cam0` | 1,660 | 20.0% | θ = 0°, φ = 0° | Frontal view |
| `cam1` | 1,660 | 20.0% | θ = -30°, φ = 0° | Left horizontal angle |
| `cam2` | 1,660 | 20.0% | θ = +30°, φ = 0° | Right horizontal angle |
| `cam3` | 1,660 | 20.0% | θ = 0°, φ = +15° | Elevated vertical angle |
| `cam4` | 1,660 | 20.0% | θ = 0°, φ = -15° | Depressed vertical angle |

---

## 4. Unicode & Label Parser Findings
1. **Whitespace Inconsistencies**:
   - Original filenames contained trailing spaces on select signs (e.g., `'أمتعة .mp4'` and `'اسم .mp4'`).
   - `LabelParser` strips trailing whitespace while mapping back to exact disk paths.
2. **Numeric Signs**:
   - 21 classes are numeric values (`1`..`10`, `16`..`20`, `30`, `40`, `50`, `70`, `80`, `90`).
   - Sorted numerically (1, 2, ..., 10, 16) rather than lexicographically ("1", "10", "2").
3. **Multi-word Signs**:
   - 24 signs consist of compound terms (e.g., `طبيب أسنان`, `إعلان عام`, `سيرا على الأقدام`).
4. **Encoding Standard**:
   - UTF-8 with NFC normalization is enforced across all manifest outputs to prevent `cp1252` encoding corruption on Windows.

---

## 5. Corrupted or Excluded Files Report
No corrupted, unreadable, or zero-frame video files were detected. All 8,300 files decoded successfully.

---

## 6. License and Attribution
- **Published License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Commercial Use**: Permitted under CC BY 4.0 provided appropriate credit is given and license link is provided.
- **Reference**:
  > Younes Ouargani, Noussaima El Khattabi. "VDzSL: A synthetic video dataset for Algerian sign language using 3D avatars." *Data in Brief*, 2026, 112551. DOI: `10.1016/j.dib.2026.112551`.
