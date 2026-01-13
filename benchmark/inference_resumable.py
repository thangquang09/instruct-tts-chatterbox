"""
Resumable Benchmark Inference for Instruction TTS.

This script:
1. Loads test metadata from TXT file (pipe-separated: audio_name|text|instruction)
2. Checks which files are already processed in output directory
3. Only processes remaining files
4. Names output files by audio_name (without .wav extension)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchaudio
from chatterbox.tts import InstructionChatterBox


def get_processed_ids(output_dir: Path) -> set:
    """Get set of already processed IDs from output directory."""
    processed = set()
    if output_dir.exists():
        for wav_file in output_dir.glob("*.wav"):
            # ID is the filename without extension
            processed.add(wav_file.stem)
    return processed


def sanitize_id(raw_id: str) -> str:
    """Sanitize ID for use as filename.

    Replaces characters that are invalid in filenames (like '/') with '_'.
    Example:
        test-clean/8555/284447/8555_284447_000020_000002
        -> test-clean_8555_284447_8555_284447_000020_000002
    """
    return raw_id.replace("/", "_").replace("\\", "_")


def load_test_cases_from_txt(txt_path: str) -> list:
    """Load test cases from pipe-separated TXT file.

    Format: audio_name|text|instruction
    Example: p326_358.wav|Labour is providing none of these.|A mature, Australian male voice...

    Uses sanitized audio_name (without .wav, with '/' replaced by '_') as unique ID.
    """
    test_cases = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line[:50]}...")
                continue
            audio_name, text, instruction = parts
            # Use audio_name without .wav, sanitized for filename safety
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


def main():
    parser = argparse.ArgumentParser(
        description="Resumable Benchmark Inference for Instruction TTS"
    )
    parser.add_argument(
        "--part",
        type=int,
        required=True,
        help="Part index (0 to total_parts-1) to process",
    )
    parser.add_argument(
        "--total_parts",
        type=int,
        default=9,
        help="Total number of parts to split data into (default: 9 for 3 GPUs x 3 processes)",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="2query_t3_freeze",
        help="Subdirectory under benchmark/output_wavs/ for outputs",
    )
    parser.add_argument(
        "--t3_ckpt_dir",
        type=str,
        default="checkpoints/t3_instruct_ddp_2query",
        help="Path to T3 checkpoint directory",
    )
    args = parser.parse_args()

    # Paths
    TXT_PATH = "data/full_test.txt"
    MAPPER_CKPT = "checkpoints/mapper_flow/best_model.pt"
    OUTPUT_DIR = Path(f"benchmark/output_wavs/{args.output_subdir}/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # =================== Check Already Processed ===================
    processed_ids = get_processed_ids(OUTPUT_DIR)
    print(f"Found {len(processed_ids)} already processed files in {OUTPUT_DIR}")

    # =================== Load All Test Cases ===================
    if not os.path.exists(TXT_PATH):
        raise FileNotFoundError(f"Test data not found: {TXT_PATH}")

    all_test_cases = load_test_cases_from_txt(TXT_PATH)
    total_samples = len(all_test_cases)
    print(f"Total samples in TXT: {total_samples}")

    # =================== Split for Multi-GPU (FIXED: Split on ALL data, not remaining) ===================
    # IMPORTANT: Split based on ALL test cases to ensure each part always gets
    # the same subset, regardless of how many times the script is restarted.
    # This prevents samples from being missed or duplicated when processes restart.
    part_size = len(all_test_cases) // args.total_parts
    start_idx = args.part * part_size

    # Last part takes all remaining samples
    if args.part == args.total_parts - 1:
        end_idx = len(all_test_cases)
    else:
        end_idx = start_idx + part_size

    # Get this part's fixed subset from ALL test cases
    my_cases_all = all_test_cases[start_idx:end_idx]

    # NOW filter out already processed samples from this part's subset
    my_cases = [tc for tc in my_cases_all if tc["id"] not in processed_ids]

    # Also calculate global remaining for display
    total_remaining = len(
        [tc for tc in all_test_cases if tc["id"] not in processed_ids]
    )

    print(f"\n{'=' * 60}")
    print(f"Part {args.part}/{args.total_parts - 1}")
    print(
        f"Assigned samples: {len(my_cases_all)} (indices {start_idx} to {end_idx - 1} of all)"
    )
    print(f"Already processed in this part: {len(my_cases_all) - len(my_cases)}")
    print(f"Remaining to process in this part: {len(my_cases)}")
    print(f"Total remaining globally: {total_remaining}")
    print(f"{'=' * 60}")

    if not my_cases:
        print("No samples to process for this part. Exiting.")
        return

    # =================== Load Model ===================
    print("\nLoading model...")
    model = InstructionChatterBox.from_local(
        t3_ckpt_dir=args.t3_ckpt_dir,
        mapper_ckpt_path=MAPPER_CKPT,
        device=DEVICE,
    )

    # =================== Generate Speech ===================
    print("\n" + "=" * 60)
    print("Generating speech samples...")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for i, case in enumerate(my_cases):
        print(f"\n[{i + 1}/{len(my_cases)}] ID: {case['id']}")
        print(f"  Text: {case['text'][:50]}...")
        print(f"  Instruction: {case['instruction'][:60]}...")

        output_path = OUTPUT_DIR / f"{case['id']}.wav"

        # Double-check (in case another process generated it)
        if output_path.exists():
            print(f"  ⏭ Already exists, skipping: {output_path}")
            continue

        try:
            # Generate
            wav = model.generate(
                text=case["text"],
                instruction=case["instruction"],
                temperature=1.0,
                cfg_weight=0.3,
            )

            # Save with ID as filename
            torchaudio.save(
                str(output_path),
                wav.cpu(),
                sample_rate=model.sr,
            )
            print(f"  ✓ Saved to: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            error_count += 1

    print("\n" + "=" * 60)
    print(f"Part {args.part} Done!")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
