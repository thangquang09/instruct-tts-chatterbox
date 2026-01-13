"""
Merge evaluation results from prediction CSVs and groundtruth CSVs.

Merges:
- Predictions:
  - run_base_eval.py output: {subdir}_base.csv
  - run_emotion_accent_eval.py output: {subdir}_emotion_accent.csv
- Groundtruth:
  - gt_tags.csv (pitch, monotony, speed, age, gender)
  - gt_emotion_accent.csv (emotion, accent)

Handles:
- Predictions may have more samples than groundtruth (20k vs 17.9k)
- Groundtruth uses full audio_path, predictions use sample ID
- Left join on predictions - groundtruth columns will be NaN for extra samples

Usage:
    python merge_eval_results.py \
        --pred_base benchmark/results/2query_t3_unfreeze_base.csv \
        --pred_emotion_accent benchmark/results/2query_t3_unfreeze_emotion_accent.csv \
        --gt_tags benchmark/results/gt_tags.csv \
        --gt_emotion_accent benchmark/results/gt_emotion_accent.csv \
        --output benchmark/results/2query_t3_unfreeze_full.csv
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def sanitize_id(raw_id: str) -> str:
    """Sanitize ID for use as filename.

    Replaces characters that are invalid in filenames (like '/') with '_'.
    """
    return raw_id.replace("/", "_").replace("\\", "_")


def extract_id_from_audio_path(audio_path: str) -> str:
    """Extract sample ID from full audio path.

    Example:
        /data1/.../VCTK-Corpus/wav48/p326/p326_358.wav -> p326_358
        gpt4o_19981_surprised_verse.wav -> gpt4o_19981_surprised_verse
        test-clean/8555/284447/8555_284447_000020_000002.wav
            -> test-clean_8555_284447_8555_284447_000020_000002
    """
    filename = os.path.basename(audio_path)  # p326_358.wav
    raw_id = filename.replace(".wav", "")  # p326_358
    sample_id = sanitize_id(raw_id)
    return sample_id


def load_and_add_id(csv_path: str, sep: str = "|") -> pd.DataFrame:
    """Load CSV and add 'id' column extracted from audio_path."""
    df = pd.read_csv(csv_path, sep=sep)
    if "audio_path" in df.columns and "id" not in df.columns:
        df["id"] = df["audio_path"].apply(extract_id_from_audio_path)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge prediction and groundtruth evaluation results"
    )
    parser.add_argument(
        "--pred_base",
        type=str,
        required=True,
        help="Path to base prediction CSV (from run_base_eval.py)",
    )
    parser.add_argument(
        "--pred_emotion_accent",
        type=str,
        required=True,
        help="Path to emotion/accent prediction CSV (from run_emotion_accent_eval.py)",
    )
    parser.add_argument(
        "--gt_tags",
        type=str,
        default="benchmark/results/gt_tags.csv",
        help="Path to groundtruth tags CSV (pitch, monotony, speed, age, gender)",
    )
    parser.add_argument(
        "--gt_emotion_accent",
        type=str,
        default="benchmark/results/gt_emotion_accent.csv",
        help="Path to groundtruth emotion/accent CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output merged CSV",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MERGING PREDICTION AND GROUNDTRUTH RESULTS")
    print("=" * 60)

    # ===== LOAD PREDICTION RESULTS =====
    print("\n--- Loading Prediction Results ---")

    # Base predictions
    print(f"[1/4] Loading base predictions: {args.pred_base}")
    df_pred_base = pd.read_csv(args.pred_base, sep="|")
    print(f"      Samples: {len(df_pred_base)}, Columns: {list(df_pred_base.columns)}")

    # Emotion/Accent predictions
    print(f"[2/4] Loading emotion/accent predictions: {args.pred_emotion_accent}")
    df_pred_ea = pd.read_csv(args.pred_emotion_accent, sep="|")
    print(f"      Samples: {len(df_pred_ea)}, Columns: {list(df_pred_ea.columns)}")

    # ===== LOAD GROUNDTRUTH RESULTS =====
    print("\n--- Loading Groundtruth Results ---")

    # GT Tags
    print(f"[3/4] Loading groundtruth tags: {args.gt_tags}")
    df_gt_tags = load_and_add_id(args.gt_tags)
    print(f"      Samples: {len(df_gt_tags)}, Columns: {list(df_gt_tags.columns)}")

    # GT Emotion/Accent
    print(f"[4/4] Loading groundtruth emotion/accent: {args.gt_emotion_accent}")
    df_gt_ea = load_and_add_id(args.gt_emotion_accent)
    print(f"      Samples: {len(df_gt_ea)}, Columns: {list(df_gt_ea.columns)}")

    # ===== MERGE PREDICTIONS =====
    print("\n--- Merging Predictions ---")

    # Select only prediction columns from emotion_accent (avoid duplicating id, audio_name)
    pred_ea_cols = ["id", "pred_emotion", "pred_accent"]
    df_pred_ea_subset = df_pred_ea[pred_ea_cols]

    # Merge base + emotion_accent predictions
    df_merged = pd.merge(df_pred_base, df_pred_ea_subset, on="id", how="left")
    print(f"After merging predictions: {len(df_merged)} samples")

    # ===== MERGE GROUNDTRUTH =====
    print("\n--- Merging Groundtruth (left join - predictions as base) ---")

    # Prepare GT tags (select only needed columns)
    gt_tags_cols = [
        "id",
        "gt_pitch",
        "gt_speech_monotony",
        "gt_speaking_rate",
        "gt_age",
        "gt_gender",
    ]
    gt_tags_cols = [c for c in gt_tags_cols if c in df_gt_tags.columns]
    df_gt_tags_subset = df_gt_tags[gt_tags_cols]

    # Prepare GT emotion/accent (select only needed columns)
    gt_ea_cols = ["id", "gt_emotion", "gt_accent"]
    gt_ea_cols = [c for c in gt_ea_cols if c in df_gt_ea.columns]
    df_gt_ea_subset = df_gt_ea[gt_ea_cols]

    # Merge GT tags
    df_merged = pd.merge(df_merged, df_gt_tags_subset, on="id", how="left")
    print(f"After merging GT tags: {len(df_merged)} samples")

    # Merge GT emotion/accent
    df_merged = pd.merge(df_merged, df_gt_ea_subset, on="id", how="left")
    print(f"After merging GT emotion/accent: {len(df_merged)} samples")

    # ===== STATISTICS =====
    print("\n--- Merge Statistics ---")
    total_samples = len(df_merged)

    # Count samples with groundtruth
    has_gt_tags = (
        df_merged["gt_pitch"].notna().sum() if "gt_pitch" in df_merged.columns else 0
    )
    has_gt_emotion = (
        df_merged["gt_emotion"].notna().sum()
        if "gt_emotion" in df_merged.columns
        else 0
    )

    print(f"  Total samples: {total_samples}")
    print(
        f"  Samples with GT tags: {has_gt_tags} ({100 * has_gt_tags / total_samples:.1f}%)"
    )
    print(
        f"  Samples with GT emotion/accent: {has_gt_emotion} ({100 * has_gt_emotion / total_samples:.1f}%)"
    )
    print(f"  Samples without GT: {total_samples - has_gt_tags}")

    # ===== REORDER COLUMNS =====
    # Order: id, audio_name, text, instruction, pred_*, gt_*
    pred_cols = [c for c in df_merged.columns if c.startswith("pred_")]
    gt_cols = [c for c in df_merged.columns if c.startswith("gt_")]
    other_cols = [
        c for c in df_merged.columns if c not in pred_cols and c not in gt_cols
    ]

    ordered_cols = other_cols + pred_cols + gt_cols
    df_merged = df_merged[ordered_cols]

    # ===== SAVE =====
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, sep="|", index=False)

    print(f"\n--- Results saved to: {output_path} ---")
    print(f"Final columns: {list(df_merged.columns)}")

    # ===== PRINT SUMMARY METRICS (only for samples with GT) =====
    print("\n" + "=" * 60)
    print("SUMMARY METRICS (Predictions)")
    print("=" * 60)

    if "pred_wer" in df_merged.columns:
        print(f"  Average WER: {df_merged['pred_wer'].mean():.4f}")
    if "pred_cer" in df_merged.columns:
        print(f"  Average CER: {df_merged['pred_cer'].mean():.4f}")
    if "pred_utmos" in df_merged.columns:
        print(f"  Average UTMOS: {df_merged['pred_utmos'].mean():.4f}")

    # ===== ACCURACY (for samples with GT) =====
    print("\n" + "=" * 60)
    print("ACCURACY (Samples with Groundtruth)")
    print("=" * 60)

    df_with_gt = (
        df_merged[df_merged["gt_pitch"].notna()]
        if "gt_pitch" in df_merged.columns
        else df_merged
    )

    if len(df_with_gt) > 0:
        # Gender accuracy
        if "pred_gender" in df_with_gt.columns and "gt_gender" in df_with_gt.columns:
            gender_acc = (df_with_gt["pred_gender"] == df_with_gt["gt_gender"]).mean()
            print(f"  Gender Accuracy: {gender_acc:.4f}")

        # Age accuracy
        if "pred_age" in df_with_gt.columns and "gt_age" in df_with_gt.columns:
            age_acc = (df_with_gt["pred_age"] == df_with_gt["gt_age"]).mean()
            print(f"  Age Accuracy: {age_acc:.4f}")

        # Pitch accuracy
        if "pred_pitch" in df_with_gt.columns and "gt_pitch" in df_with_gt.columns:
            pitch_acc = (df_with_gt["pred_pitch"] == df_with_gt["gt_pitch"]).mean()
            print(f"  Pitch Accuracy: {pitch_acc:.4f}")

        # Speed accuracy
        if (
            "pred_speaking_rate" in df_with_gt.columns
            and "gt_speaking_rate" in df_with_gt.columns
        ):
            speed_acc = (
                df_with_gt["pred_speaking_rate"] == df_with_gt["gt_speaking_rate"]
            ).mean()
            print(f"  Speaking Rate Accuracy: {speed_acc:.4f}")

        # Monotony accuracy
        if (
            "pred_speech_monotony" in df_with_gt.columns
            and "gt_speech_monotony" in df_with_gt.columns
        ):
            mono_acc = (
                df_with_gt["pred_speech_monotony"] == df_with_gt["gt_speech_monotony"]
            ).mean()
            print(f"  Speech Monotony Accuracy: {mono_acc:.4f}")

    df_with_gt_ea = (
        df_merged[df_merged["gt_emotion"].notna()]
        if "gt_emotion" in df_merged.columns
        else df_merged
    )

    if len(df_with_gt_ea) > 0:
        # Emotion accuracy
        if (
            "pred_emotion" in df_with_gt_ea.columns
            and "gt_emotion" in df_with_gt_ea.columns
        ):
            emotion_acc = (
                df_with_gt_ea["pred_emotion"] == df_with_gt_ea["gt_emotion"]
            ).mean()
            print(f"  Emotion Accuracy: {emotion_acc:.4f}")

        # Accent accuracy
        if (
            "pred_accent" in df_with_gt_ea.columns
            and "gt_accent" in df_with_gt_ea.columns
        ):
            accent_acc = (
                df_with_gt_ea["pred_accent"] == df_with_gt_ea["gt_accent"]
            ).mean()
            print(f"  Accent Accuracy: {accent_acc:.4f}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
