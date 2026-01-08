#!/usr/bin/env python3
"""
Example script to test InstructionChatterBox - instruction-only TTS.

This script demonstrates generating speech using only:
- Text content to synthesize
- Text instruction describing the desired voice/style

NO reference audio is required!

Usage:
    python example_instruction_tts.py

Requirements:
    - Trained InstructionMapper checkpoint (mapper_slice_v2/best_model.pt)
    - Finetuned T3 checkpoint (t3_instruct_ddp/)
"""

import os
import shutil
import torch
import torchaudio
from pathlib import Path

# Add src to path if running from project root
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts import InstructionChatterBox


def main():
    # =================== Configuration ===================

    # Checkpoint paths
    T3_CKPT_DIR = "./checkpoints/t3_instruct_ddp"
    MAPPER_CKPT = "checkpoints/mapper_flow/best_model.pt"

    # Valid directory
    VALID_DIR = "data/final_data_test.txt"

    # Output directory
    OUTPUT_DIR = Path("./outputs/instruction_tts_test/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # =================== Load Model ===================

    print("\n" + "=" * 60)
    print("Loading InstructionChatterBox...")
    print("=" * 60)

    model = InstructionChatterBox.from_local(
        t3_ckpt_dir=T3_CKPT_DIR,
        mapper_ckpt_path=MAPPER_CKPT,
        device=DEVICE,
    )

    # =================== Load Data ===================

    if not os.path.exists(VALID_DIR):
        raise FileNotFoundError(f"Không tìm thấy file data tại: {VALID_DIR}")

    with open(VALID_DIR, "r", encoding="utf-8") as f:  # Thêm encoding utf-8 cho an toàn
        lines = f.readlines()

    # tạo folder output
    GT_PATH = OUTPUT_DIR / "gt"  # Dùng Path object tiện hơn
    GT_PATH.mkdir(parents=True, exist_ok=True)

    test_cases = []
    # random 10 samples trong lines
    import random

    random.shuffle(lines)
    lines = lines[:10]

    for line in lines:
        # .strip() để loại bỏ \n ở đầu cuối
        parts = line.strip().split("|")

        # Kiểm tra xem dòng có đủ 3 phần không để tránh lỗi index
        if len(parts) >= 3:
            test_cases.append(
                {
                    "gt_path": parts[0],
                    "text": parts[1],
                    "instruction": parts[2],
                    # Lấy tên file an toàn hơn bằng Path
                    "output_name": Path(parts[0]).stem,
                }
            )

    # =================== Generate Speech ===================

    print("\n" + "=" * 60)
    print("Generating speech samples...")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        print(f"\n[{i + 1}/{len(test_cases)}] {case['output_name']}")
        print(f"  Text: {case['text'][:50]}...")
        print(f"  Instruction: {case['instruction']}")

        try:
            # Generate
            wav = model.generate(
                text=case["text"],
                instruction=case["instruction"],
            )

            # Save
            output_path = OUTPUT_DIR / f"{case['output_name']}.wav"
            torchaudio.save(
                str(output_path),
                wav.cpu(),
                sample_rate=model.sr,
            )
            print(f"  ✓ Gen Saved to: {output_path}")

            # Copy Ground Truth (GT)
            src_gt_path = Path(case["gt_path"])
            dst_gt_path = GT_PATH / f"{case['output_name']}.wav"

            if src_gt_path.exists():
                shutil.copy(src_gt_path, dst_gt_path)
                print(f"  ✓ GT Copied to: {dst_gt_path}")
            else:
                print(f"  ! GT file not found at source: {src_gt_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Done! Check outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
