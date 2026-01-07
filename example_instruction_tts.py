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
    T3_CKPT_DIR = "checkpoints/t3_instruct_ddp"
    MAPPER_CKPT = "checkpoints/mapper_slice_v4/best_model.pt"

    # Output directory
    OUTPUT_DIR = Path("./outputs/instruction_tts")
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

    # =================== Test Cases ===================
    instructions = [
        # --- GROUP 1: CẢM XÚC ĐỐI LẬP ---
        "A young woman speaking in a very happy, excited, and laughing tone.",
        "A sad man, speaking slowly with a depressed and crying voice.",
        "An angry male voice, shouting and aggressive.",
        "A scared young girl, whispering and trembling with fear.",
        # --- GROUP 2: CAO ĐỘ & ĐỘ TUỔI ---
        "A man with a very deep, bass-heavy, and dominant voice.",
        "A cute little boy with a very high-pitched and squeaky voice.",
        "An old grandfather with a raspy, shaky, and tired voice.",
        # --- GROUP 3: PHONG CÁCH ĐẶC BIỆT ---
        "A mysterious woman whispering softly into the microphone.",
        "A robotic voice, monotone, flat, and without emotion.",
        "A very fast-talking and energetic salesperson.",
        "A very slow, sleepy, and yawning voice.",
    ]

    output_names = [
        # --- GROUP 1 ---
        "young_woman_happy_laughing",
        "man_sad_depressed_crying",
        "male_angry_shouting",
        "girl_scared_whispering",
        # --- GROUP 2 ---
        "man_deep_bass_dominant",
        "boy_high_pitch_squeaky",
        "grandfather_raspy_shaky",
        # --- GROUP 3 ---
        "woman_mysterious_whisper_asmr",
        "robotic_monotone_flat",
        "salesperson_fast_energetic",
        "voice_slow_sleepy_yawning",
    ]

    text = "Pig farmers are completely despondent."

    test_cases = [
        {
            "text": text,
            "instruction": instructions[i],
            "output_name": output_names[i],
        }
        for i in range(len(instructions))
    ]

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
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
            )

            # Save
            output_path = OUTPUT_DIR / f"{case['output_name']}.wav"
            torchaudio.save(
                str(output_path),
                wav,
                sample_rate=model.sr,
            )
            print(f"  ✓ Saved to: {output_path}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Done! Check outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
