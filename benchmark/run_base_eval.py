"""
Base Evaluation Script for Generated TTS Audio.

Evaluates pre-generated audio files for:
- Pitch (pitch level)
- Speech monotony (expressiveness)
- Speaking rate (speed)
- Age estimation
- Gender detection
- WER/CER (using Whisper ASR)
- UTMOS (audio quality)

Input:
- TXT file: audio_name|text|instruction (from data/full_test.txt)
- WAV files: pre-generated audio from benchmark/output_wavs/{subdir}/

Output:
- CSV with evaluation results
"""

import os
import sys
import json
import bisect
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import whisper
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer as calculate_wer
from jiwer import cer as calculate_cer
import utmosv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
BASE_DIR = Path(__file__).parent.absolute()

# Imports
from base_eval import pitch_apply, speed_apply, age_gender_apply

# =================== LABEL CONSTANTS ===================
SPEAKER_RATE_BINS = [
    "very slowly",
    "slowly",
    "slightly slowly",
    "moderate speed",
    "slightly fast",
    "fast",
    "very fast",
]
UTTERANCE_LEVEL_STD = [
    "very monotone",
    "monotone",
    "slightly expressive and animated",
    "expressive and animated",
    "very expressive and animated",
]
SPEAKER_LEVEL_PITCH_BINS = [
    "very low-pitch",
    "low-pitch",
    "slightly low-pitch",
    "moderate pitch",
    "slightly high-pitch",
    "high-pitch",
    "very high-pitch",
]

# Load text bins for base_eval
with open(os.path.join(BASE_DIR, "bin.json")) as json_file:
    TEXT_BINS_DICT = json.load(json_file)


# =================== BASE EVAL FUNCTIONS ===================
def get_pitch_label(pitch_mean, gender):
    if gender == "male":
        index = bisect.bisect_right(TEXT_BINS_DICT["pitch_bins_male"], pitch_mean) - 1
    else:
        index = bisect.bisect_right(TEXT_BINS_DICT["pitch_bins_female"], pitch_mean) - 1
    index = max(0, min(index, len(SPEAKER_LEVEL_PITCH_BINS) - 1))
    return SPEAKER_LEVEL_PITCH_BINS[index]


def get_monotony_label(pitch_std):
    index = bisect.bisect_right(TEXT_BINS_DICT["speech_monotony"], pitch_std) - 1
    index = max(0, min(index, len(UTTERANCE_LEVEL_STD) - 1))
    return UTTERANCE_LEVEL_STD[index]


def get_speed_label(speech_duration):
    index = bisect.bisect_right(TEXT_BINS_DICT["speaking_rate"], speech_duration) - 1
    index = max(0, min(index, len(SPEAKER_RATE_BINS) - 1))
    return SPEAKER_RATE_BINS[index]


def process_base_eval(waveform_np):
    """Process base evaluation metrics from numpy waveform (16kHz)."""
    age, gender = age_gender_apply(waveform_np)
    pitch_mean, pitch_std = pitch_apply(waveform_np)
    speech_duration = speed_apply(waveform_np)

    pitch_label = get_pitch_label(pitch_mean, gender)
    monotony_label = get_monotony_label(pitch_std)
    speed_label = get_speed_label(speech_duration)

    return {
        "pred_pitch": pitch_label,
        "pred_speech_monotony": monotony_label,
        "pred_speaking_rate": speed_label,
        "pred_age": age,
        "pred_gender": gender,
    }


# =================== WER FUNCTIONS ===================
def calculate_wer_metrics(waveform_np, gt_text, whisper_model, normalizer, device):
    """Calculate WER/CER from numpy waveform."""
    result = whisper_model.transcribe(
        waveform_np, fp16=(device.type == "cuda"), language="en"
    )
    pred_text = normalizer(result["text"].strip())
    gt_text_normalized = normalizer(gt_text)

    wer_score = round(calculate_wer(gt_text_normalized, pred_text), 4)
    cer_score = round(calculate_cer(gt_text_normalized, pred_text), 4)

    return {
        "pred_asr_text": pred_text,
        "pred_wer": wer_score,
        "pred_cer": cer_score,
    }


# =================== UTMOS FUNCTIONS ===================
def calculate_utmos_score(waveform_np, sample_rate, utmos_model):
    """Calculate UTMOS from numpy waveform."""
    mos_result = utmos_model.predict(data=waveform_np, sr=sample_rate)

    if isinstance(mos_result, (torch.Tensor, np.ndarray)):
        mos_score = float(np.array(mos_result).squeeze())
    else:
        mos_score = float(mos_result)

    return round(mos_score, 4)


# =================== DATA LOADING ===================
def load_test_cases_from_txt(txt_path: str) -> list:
    """Load test cases from pipe-separated TXT file."""
    test_cases = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            audio_name, text, instruction = parts
            sample_id = audio_name.replace(".wav", "")
            test_cases.append(
                {
                    "id": sample_id,
                    "audio_name": audio_name,
                    "text": text,
                    "instruction": instruction,
                }
            )
    return test_cases


# =================== MAIN ===================
def main():
    parser = argparse.ArgumentParser(
        description="Base Evaluation for Generated TTS Audio"
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        required=True,
        help="Directory containing generated WAV files (e.g., benchmark/output_wavs/2query_t3_unfreeze)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output CSV name (without path). Default: {wav_dir_name}_base.csv",
    )
    parser.add_argument(
        "--input_txt",
        type=str,
        default="data/full_test.txt",
        help="Input TXT file with test cases",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WAV_DIR = Path(args.wav_dir)
    INPUT_TXT = args.input_txt

    # Output path
    os.makedirs(BASE_DIR / "results", exist_ok=True)
    if args.output_name:
        OUTPUT_CSV = BASE_DIR / "results" / args.output_name
    else:
        OUTPUT_CSV = BASE_DIR / "results" / f"{WAV_DIR.name}_base.csv"

    if not str(OUTPUT_CSV).endswith(".csv"):
        OUTPUT_CSV = Path(str(OUTPUT_CSV) + ".csv")

    print("=" * 60)
    print("BASE EVALUATION FOR GENERATED TTS AUDIO")
    print("=" * 60)
    print(f"  WAV Directory: {WAV_DIR}")
    print(f"  Input TXT:     {INPUT_TXT}")
    print(f"  Output CSV:    {OUTPUT_CSV}")
    print(f"  Device:        {DEVICE}")
    print("=" * 60)

    # ===== CHECK WAV DIRECTORY =====
    if not WAV_DIR.exists():
        print(f"ERROR: WAV directory not found: {WAV_DIR}")
        return

    # ===== LOAD MODELS =====
    print("\n--- Loading Models ---")

    # 1. Whisper Model (for WER)
    print("[1/2] Loading Whisper Model...")
    whisper_model = whisper.load_model("large-v3-turbo", device=DEVICE)
    normalizer = EnglishTextNormalizer()

    # 2. UTMOS Model
    print("[2/2] Loading UTMOS Model...")
    utmos_model = utmosv2.create_model(pretrained=True, device=str(DEVICE))

    # 3. Base eval models are loaded globally in base_eval.py
    print("[OK] Base Eval Models loaded from base_eval.py")

    print("\n--- All Models Loaded ---")

    # ===== READ INPUT DATA =====
    print("\n--- Reading Input Data ---")
    all_test_cases = load_test_cases_from_txt(INPUT_TXT)
    print(f"Total samples in TXT: {len(all_test_cases)}")

    # ===== RESUMABLE LOGIC =====
    processed_ids = set()
    if OUTPUT_CSV.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_CSV, sep="|")
            processed_ids = set(existing_df["id"].unique())
            print(f"Found existing results. Already processed: {len(processed_ids)}")
        except Exception as e:
            print(f"Note: Existing file empty or error, starting fresh. Detail: {e}")

    cases_to_process = [tc for tc in all_test_cases if tc["id"] not in processed_ids]

    # Filter by available WAV files
    available_cases = []
    missing_count = 0
    for tc in cases_to_process:
        wav_path = WAV_DIR / f"{tc['id']}.wav"
        if wav_path.exists():
            available_cases.append(tc)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} WAV files not found, will be skipped")

    if len(available_cases) == 0:
        print("--- No samples to process! ---")
        return

    print(f"Processing {len(available_cases)} samples...")

    # ===== PROCESSING LOOP =====
    # Resampler to 16kHz
    resampler_cache = {}

    with torch.inference_mode():
        for tc in tqdm(available_cases, desc="Evaluating"):
            sample_id = tc["id"]
            text = tc["text"]
            instruction = tc["instruction"]
            wav_path = WAV_DIR / f"{sample_id}.wav"

            try:
                # 1. Load audio
                waveform, sr = torchaudio.load(str(wav_path))  # [channels, samples]

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # 2. Resample to 16kHz if needed
                if sr != 16000:
                    if sr not in resampler_cache:
                        resampler_cache[sr] = T.Resample(sr, 16000)
                    waveform = resampler_cache[sr](waveform)

                # Convert to numpy (16kHz, mono)
                wav_16k_np = waveform.squeeze(0).numpy()

                # 3. Base Eval
                base_eval_results = process_base_eval(wav_16k_np)

                # 4. WER/CER
                wer_results = calculate_wer_metrics(
                    wav_16k_np, text, whisper_model, normalizer, DEVICE
                )

                # 5. UTMOS
                pred_utmos = calculate_utmos_score(wav_16k_np, 16000, utmos_model)

                # ===== COMBINE RESULTS =====
                res_dict = {
                    "id": sample_id,
                    "audio_name": tc["audio_name"],
                    "text": text,
                    "instruction": instruction,
                    # Base Eval
                    **base_eval_results,
                    # WER
                    **wer_results,
                    # UTMOS
                    "pred_utmos": pred_utmos,
                }

                # ===== SAVE (Append mode) =====
                res_df = pd.DataFrame([res_dict])
                file_exists = OUTPUT_CSV.exists()
                res_df.to_csv(
                    OUTPUT_CSV,
                    mode="a",
                    index=False,
                    sep="|",
                    header=not file_exists,
                )

            except Exception as e:
                print(f"\nError processing {sample_id}: {e}")
                import traceback

                traceback.print_exc()

    # ===== FINAL STATISTICS =====
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)

    if OUTPUT_CSV.exists():
        final_df = pd.read_csv(OUTPUT_CSV, sep="|")
        print(f"Total samples in results: {len(final_df)}")
        print(f"Results saved to: {OUTPUT_CSV}")

        # Print average metrics
        print("\n--- Average Metrics ---")
        if "pred_wer" in final_df.columns:
            print(f"  Average WER: {final_df['pred_wer'].mean():.4f}")
        if "pred_cer" in final_df.columns:
            print(f"  Average CER: {final_df['pred_cer'].mean():.4f}")
        if "pred_utmos" in final_df.columns:
            print(f"  Average UTMOS: {final_df['pred_utmos'].mean():.4f}")


if __name__ == "__main__":
    main()
