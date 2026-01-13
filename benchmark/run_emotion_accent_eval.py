"""
Emotion & Accent Evaluation Script for Generated TTS Audio.

Evaluates pre-generated audio files for:
- Emotion classification
- Accent classification

Input:
- TXT file: audio_name|text|instruction (from data/full_test.txt)
- WAV files: pre-generated audio from benchmark/output_wavs/{subdir}/

Output:
- CSV with emotion and accent predictions

Note: This script requires a separate environment with emotion/accent model dependencies.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
BASE_DIR = Path(__file__).parent.absolute()

# Imports for emotion/accent models
from src.model.emotion.whisper_emotion import WhisperWrapper
from src.model.accent.wavlm_accent import WavLMWrapper

# =================== LABEL CONSTANTS ===================
EMOTION_LIST = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
    "Other",
]
ACCENT_LIST = [
    "East Asia",
    "English",
    "Germanic",
    "Irish",
    "North America",
    "Northern Irish",
    "Oceania",
    "Other",
    "Romance",
    "Scottish",
    "Semitic",
    "Slavic",
    "South African",
    "Southeast Asia",
    "South Asia",
    "Welsh",
]


# =================== AUDIO PROCESSING ===================
def prepare_audio_tensor(waveform_np, device, target_sr=16000, max_duration=15):
    """Prepare audio tensor for emotion/accent models."""
    waveform = torch.from_numpy(waveform_np).float().to(device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    max_length = int(target_sr * max_duration)
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]

    return waveform


# =================== PREDICTION FUNCTIONS ===================
def get_emotion(data, emotion_model):
    """Get emotion prediction from audio tensor."""
    logits, _, _, _, _, _ = emotion_model(data, return_feature=True)
    prob = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(prob).detach().cpu().item()
    return EMOTION_LIST[pred_idx]


def get_accent(data, accent_model):
    """Get accent prediction from audio tensor."""
    logits, _ = accent_model(data, return_feature=True)
    prob = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(prob).detach().cpu().item()
    return ACCENT_LIST[pred_idx]


# =================== DATA LOADING ===================
def sanitize_id(raw_id: str) -> str:
    """Sanitize ID for use as filename.

    Replaces characters that are invalid in filenames (like '/') with '_'.
    """
    return raw_id.replace("/", "_").replace("\\", "_")


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
            raw_id = audio_name.replace(".wav", "")
            sample_id = sanitize_id(raw_id)
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
        description="Emotion & Accent Evaluation for Generated TTS Audio"
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
        help="Output CSV name (without path). Default: {wav_dir_name}_emotion_accent.csv",
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
        OUTPUT_CSV = BASE_DIR / "results" / f"{WAV_DIR.name}_emotion_accent.csv"

    if not str(OUTPUT_CSV).endswith(".csv"):
        OUTPUT_CSV = Path(str(OUTPUT_CSV) + ".csv")

    print("=" * 60)
    print("EMOTION & ACCENT EVALUATION FOR GENERATED TTS AUDIO")
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

    # 1. Emotion Model
    print("[1/2] Loading Emotion Model...")
    emotion_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-msp-podcast-emotion"
    ).to(DEVICE)
    emotion_model.eval()

    # 2. Accent Model
    print("[2/2] Loading Accent Model...")
    accent_model = WavLMWrapper.from_pretrained(
        "tiantiaf/wavlm-large-narrow-accent"
    ).to(DEVICE)
    accent_model.eval()

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
        for tc in tqdm(available_cases, desc="Evaluating Emotion & Accent"):
            sample_id = tc["id"]
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

                # 3. Prepare tensor for models
                audio_tensor = prepare_audio_tensor(wav_16k_np, DEVICE)

                # 4. Emotion prediction
                pred_emotion = get_emotion(audio_tensor, emotion_model)

                # 5. Accent prediction
                pred_accent = get_accent(audio_tensor, accent_model)

                # ===== COMBINE RESULTS =====
                res_dict = {
                    "id": sample_id,
                    "audio_name": tc["audio_name"],
                    "pred_emotion": pred_emotion,
                    "pred_accent": pred_accent,
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

        # Print emotion distribution
        print("\n--- Emotion Distribution ---")
        if "pred_emotion" in final_df.columns:
            print(final_df["pred_emotion"].value_counts())

        # Print accent distribution
        print("\n--- Accent Distribution ---")
        if "pred_accent" in final_df.columns:
            print(final_df["pred_accent"].value_counts())


if __name__ == "__main__":
    main()
