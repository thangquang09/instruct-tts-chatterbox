"""
Example script for Instruct-TTS: Generate speech with style controlled by text instructions.

This script demonstrates how to use the finetuned ChatterboxTTS model that supports
text instructions to control the speaking style (e.g., "Speak happily and excited").

Usage:
    python example_instructtts.py
"""

import torchaudio as ta
import torch
from pathlib import Path

from chatterbox.tts import ChatterboxTTS


def main():
    CHECKPOINT_DIR = "./chkpt/instruct_tts_v1"
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading finetuned model from: {CHECKPOINT_DIR}")
    model = ChatterboxTTS.from_local(CHECKPOINT_DIR, device=device)
    print("Model loaded successfully!")
    



    test_text = "Hello! Today is a wonderful day and I'm so excited to share this news with you."
    
    test_instructions = [
        "A male speaker's voice is animated and high-pitched, delivering his words at a measured speed in a clean environment, conveying a happy and American tone.",
        "A female speaker with an American accent singsongedly delivers animated, happy expressions at a measured speed in a very clean environment. Her high-pitched, crisp, silky voice carries a cheerful tone, projecting loudly.",
        "A cheerful female speaker delivers an animated performance in a clean American environment, her voice characterized by a measured speed and a high-pitched tone.",
        "A male speaker with an American accent delivers confused speech at a slow speed, his high-pitched voice echoing clearly in a very clean environment.",
    ]
    
    
    
    audio_prompt_path = None
    
    print("\n" + "="*60)
    print("Generating speech with different style instructions...")
    print("="*60)
    
    for i, instruction in enumerate(test_instructions):
        print(f"\n[{i+1}/{len(test_instructions)}] Instruction: {instruction}")
        
        try:
            wav = model.generate_with_instruction(
                text=test_text,
                instruction=instruction,
                audio_prompt_path=audio_prompt_path,
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
            )
            
            output_path = OUTPUT_DIR / f"instruct_tts_sample_{i+1}.wav"
            ta.save(str(output_path), wav, model.sr)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Output files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
