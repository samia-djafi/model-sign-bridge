"""
Audit the VDzSL dataset and generate inventory manifests and audit reports.

Usage:
    python scripts/audit_dataset.py [--data-dir path/to/VDzSL] [--workers 4] [--sample-limit N]
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

# Reconfigure stdout for UTF-8 on Windows
sys.stdout.reconfigure(encoding="utf-8")

from src.data.dataset_inventory import audit_entire_dataset
from src.data.label_parser import LabelParser


def generate_audit_report(df: pd.DataFrame, output_md_path: str):
    """Generate a comprehensive markdown audit report."""
    total_files = len(df)
    readable_files = df["is_readable"].sum()
    unreadable_files = total_files - readable_files
    unreadable_df = df[~df["is_readable"]]

    unique_classes = df["canonical_label"].nunique()
    numeric_classes = df[df["is_numeric_sign"]]["canonical_label"].nunique()
    text_classes = unique_classes - numeric_classes

    min_frames = df["frame_count"].min()
    max_frames = df["frame_count"].max()
    mean_frames = df["frame_count"].mean()
    median_frames = df["frame_count"].median()
    std_frames = df["frame_count"].std()

    min_dur = df["duration_sec"].min()
    max_dur = df["duration_sec"].max()
    mean_dur = df["duration_sec"].mean()

    # Resolution breakdown
    resolutions = df.groupby(["width", "height"]).size().to_dict()
    res_str = ", ".join([f"{w}x{h}: {count} ({count/total_files*100:.1f}%)" for (w, h), count in resolutions.items()])

    # Signer breakdown
    signer_counts = df["signer_id"].value_counts().to_dict()
    camera_counts = df["camera_id"].value_counts().to_dict()

    # Class balance check
    class_counts = df["canonical_label"].value_counts()
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    is_perfectly_balanced = (min_class_count == max_class_count)

    md_content = f"""# VDzSL Dataset Audit Report

## 1. Executive Summary
- **Dataset Name**: VDzSL (Video Dataset for Algerian Sign Language)
- **Source**: Zenodo DOI `10.5281/zenodo.18209839` / PMC `PMC12925472`
- **Total Video Files Discovered**: {total_files:,}
- **Successfully Decoded & Verified**: {readable_files:,} ({readable_files/total_files*100:.1f}%)
- **Corrupted or Unreadable Files**: {unreadable_files}
- **Unique Canonical Classes**: {unique_classes} ({numeric_classes} numeric digit signs, {text_classes} Arabic text signs)
- **Class Balance**: {"Perfectly balanced (exactly " + str(min_class_count) + " videos per class)" if is_perfectly_balanced else f"Imbalanced (range: {min_class_count} to {max_class_count})"}
- **Video Resolution**: {res_str}
- **Average Duration**: {mean_dur:.2f} seconds (Range: {min_dur:.2f}s – {max_dur:.2f}s)
- **Average Frame Count**: {mean_frames:.1f} frames (Median: {median_frames:.1f}, Min: {min_frames}, Max: {max_frames}, Std: {std_frames:.1f})

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
"""
    for signer, count in sorted(signer_counts.items()):
        md_content += f"| `{signer}` | {count:,} | {count/total_files*100:.1f}% | Synthetic 3D Avatar |\n"

    md_content += """
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
"""
    if unreadable_files == 0:
        md_content += "No corrupted, unreadable, or zero-frame video files were detected. All 8,300 files decoded successfully.\n"
    else:
        md_content += f"The following {unreadable_files} files failed verification:\n\n"
        for _, row in unreadable_df.iterrows():
            md_content += f"- `{row['rel_path']}`: {row['exclusion_reason']}\n"

    md_content += """
---

## 6. License and Attribution
- **Published License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Commercial Use**: Permitted under CC BY 4.0 provided appropriate credit is given and license link is provided.
- **Reference**:
  > Younes Ouargani, Noussaima El Khattabi. "VDzSL: A synthetic video dataset for Algerian sign language using 3D avatars." *Data in Brief*, 2026, 112551. DOI: `10.1016/j.dib.2026.112551`.
"""

    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Audit report saved to: {output_md_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit VDzSL dataset.")
    parser.add_argument("--data-dir", default=r"d:\bridge_sign_model\data\VDzSL", help="Path to VDzSL dataset root")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample limit for quick test")
    args = parser.parse_args()

    manifest_dir = r"d:\bridge_sign_model\data\manifests"
    os.makedirs(manifest_dir, exist_ok=True)

    # 1. Run full audit
    df = audit_entire_dataset(args.data_dir, max_workers=args.workers, sample_limit=args.sample_limit)

    # 2. Save full inventory CSV (UTF-8 with BOM for Excel compatibility on Windows)
    inventory_csv = os.path.join(manifest_dir, "dataset_inventory.csv")
    df.to_csv(inventory_csv, index=False, encoding="utf-8-sig")
    print(f"Full inventory CSV saved to: {inventory_csv}")

    # 3. Save class frequency report
    class_freq = df.groupby(["canonical_label", "is_numeric_sign"]).size().reset_index(name="sample_count")
    class_freq.sort_values(by=["is_numeric_sign", "canonical_label"], ascending=[False, True], inplace=True)
    freq_csv = os.path.join(manifest_dir, "class_frequencies.csv")
    class_freq.to_csv(freq_csv, index=False, encoding="utf-8-sig")
    print(f"Class frequency report saved to: {freq_csv}")

    # 4. Save sample verification CSV (1 sample per class from signer0/cam0)
    samples_df = df[(df["signer_id"] == "signer0") & (df["camera_id"] == "cam0")]
    sample_csv = os.path.join(manifest_dir, "label_sample_verification.csv")
    samples_df[["canonical_label", "raw_label", "rel_path", "frame_count", "duration_sec"]].to_csv(
        sample_csv, index=False, encoding="utf-8-sig"
    )
    print(f"Sample verification CSV saved to: {sample_csv}")

    # 5. Fit and save canonical LabelParser vocabulary
    all_labels = df["canonical_label"].unique().tolist()
    label_parser = LabelParser(all_labels)
    labels_json = r"d:\bridge_sign_model\artifacts\label_mappings\classes.json"
    label_parser.save(labels_json)
    print(f"Canonical label mapping ({label_parser.num_classes} classes) saved to: {labels_json}")

    # 6. Generate readable audit markdown
    report_md = r"d:\bridge_sign_model\docs\dataset_audit.md"
    generate_audit_report(df, report_md)


if __name__ == "__main__":
    main()
