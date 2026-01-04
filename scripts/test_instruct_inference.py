#!/usr/bin/env python3
"""
Test script for instruction-only inference.
Uses InstructionMapper to generate speech without reference audio.

Usage:
    python scripts/test_instruct_inference.py --ckpt_dir ./checkpoints/t3_instruct_mock
"""

import argparse
import torch
import torchaudio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.tts import ChatterboxTTS


def main():
    parser = argparse.ArgumentParser(description="Test instruction-only TTS inference")
    parser.add_argument(
        "--ckpt_dir", 
        type=str, 
        default="./checkpoints/t3_instruct_mock",
        help="Path to checkpoint directory containing t3_cfg.safetensors and mapper.pt"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        default="Hello, this is a test of instruction-based speech synthesis.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--instruction", 
        type=str, 
        default="Speak with a deep, calm, and friendly voice.",
        help="Style instruction for voice generation"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output_instruct.wav",
        help="Output audio file path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--cfg_weight", 
        type=float, 
        default=0.5,
        help="Classifier-free guidance weight"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Instruction-Only TTS Inference Test")
    print("=" * 60)
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Device: {args.device}")
    print(f"Text: {args.text}")
    print(f"Instruction: {args.instruction}")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading model...")
    model = ChatterboxTTS.from_local(args.ckpt_dir, args.device)
    
    # Check if mapper is loaded
    if model.instruction_mapper is None:
        print("ERROR: InstructionMapper not found in checkpoint!")
        print("Make sure mapper.pt exists in the checkpoint directory.")
        return 1
    
    print(f"  -> T3 loaded: {type(model.t3)}")
    print(f"  -> InstructionMapper loaded: {type(model.instruction_mapper)}")
    print(f"  -> S3Gen loaded: {type(model.s3gen)}")
    
    # Generate audio (no audio_prompt_path -> uses mapper)
    print("\n[2/3] Generating audio (instruction-only mode)...")
    try:
        wav = model.generate_with_instruction(
            text=args.text,
            instruction=args.instruction,
            audio_prompt_path=None,  # <-- No reference audio!
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
        )
        print(f"  -> Generated waveform shape: {wav.shape}")
    except Exception as e:
        print(f"ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save audio
    print(f"\n[3/3] Saving audio to {args.output}...")
    torchaudio.save(args.output, wav, model.sr)
    print(f"  -> Saved successfully!")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Audio generated with instruction-only mode.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
