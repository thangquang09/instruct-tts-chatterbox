#!/usr/bin/env python3
"""
Preprocess and Cache Data for T3 Finetuning
=============================================
This script pre-computes expensive tokenization and embedding operations
to speed up training data loading.

Cached items per sample:
- speech_tokens: S3 Tokenizer output
- cond_prompt_tokens: First 3s prompt tokens
- speaker_emb: VoiceEncoder output (256-dim)
- instruction_ids: T5 Tokenizer output
- text: Original text (for text tokenization at load time)

Usage:
    python scripts/preprocess_cache.py \
        --metadata_file final_data.txt \
        --output_dir ./cache/t3_train \
        --num_workers 8
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

import torch
import numpy as np
import librosa
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3tokenizer import S3_SR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_models(device="cuda"):
    """Load models needed for preprocessing."""
    logger.info("Loading ChatterboxTTS model...")
    chatterbox = ChatterboxTTS.from_pretrained(device=device)
    
    logger.info("Loading T5 Tokenizer...")
    instruction_tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large", 
        use_fast=True
    )
    
    return chatterbox, instruction_tokenizer


def process_single_item(args, chatterbox, instruction_tokenizer, t3_config):
    """Process a single item and return cached data."""
    idx, audio_path, text, instruction = args
    
    try:
        # Load audio
        wav_16k, _ = librosa.load(audio_path, sr=S3_SR, mono=True)
        if len(wav_16k) == 0:
            return None, f"Empty audio: {audio_path}"
        
        # Speaker embedding
        speaker_emb_np = chatterbox.ve.embeds_from_wavs([wav_16k], sample_rate=S3_SR)
        speaker_emb = torch.from_numpy(speaker_emb_np[0]).float()
        
        # Full speech tokens
        with torch.no_grad():
            raw_tokens, lengths = chatterbox.s3gen.tokenizer.forward([wav_16k])
            if raw_tokens is None:
                return None, f"Tokenization failed: {audio_path}"
            speech_tokens = raw_tokens.squeeze(0)[:lengths.squeeze(0).item()].cpu()
        
        # Prompt tokens (first 3s)
        prompt_len_samples = int(3.0 * S3_SR)
        cond_audio = wav_16k[:prompt_len_samples]
        
        if len(cond_audio) > 0:
            with torch.no_grad():
                cond_tokens, _ = chatterbox.s3gen.tokenizer.forward(
                    [cond_audio], 
                    max_len=t3_config.speech_cond_prompt_len
                )
                if cond_tokens is not None:
                    cond_prompt_tokens = cond_tokens.squeeze(0).cpu()
                else:
                    cond_prompt_tokens = torch.zeros(t3_config.speech_cond_prompt_len, dtype=torch.long)
        else:
            cond_prompt_tokens = torch.zeros(t3_config.speech_cond_prompt_len, dtype=torch.long)
        
        # Pad/truncate cond_prompt_tokens
        target_len = t3_config.speech_cond_prompt_len
        if cond_prompt_tokens.size(0) > target_len:
            cond_prompt_tokens = cond_prompt_tokens[:target_len]
        elif cond_prompt_tokens.size(0) < target_len:
            cond_prompt_tokens = torch.nn.functional.pad(
                cond_prompt_tokens, 
                (0, target_len - cond_prompt_tokens.size(0)), 
                value=0
            )
        
        # Instruction tokens
        instruction_ids = instruction_tokenizer(
            instruction if instruction else "",
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=True
        ).input_ids.squeeze(0)
        
        cached_item = {
            "idx": idx,
            "audio_path": audio_path,
            "text": text,
            "instruction": instruction,
            "speech_tokens": speech_tokens,
            "cond_prompt_tokens": cond_prompt_tokens,
            "speaker_emb": speaker_emb,
            "instruction_ids": instruction_ids,
        }
        
        return cached_item, None
        
    except Exception as e:
        return None, f"Error processing {audio_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Preprocess and cache T3 training data")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to metadata file (format: audio|text|instruction)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save cached .pt files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference")
    parser.add_argument("--batch_save_size", type=int, default=1000,
                        help="Number of items per .pt file")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for resuming")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info(f"Loading metadata from {args.metadata_file}...")
    metadata_path = Path(args.metadata_file)
    dataset_root = metadata_path.parent
    
    items = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for idx, parts in enumerate(reader):
            if not parts or len(parts) < 2:
                continue
            audio_file = parts[0]
            text = parts[1]
            instruction = parts[2] if len(parts) > 2 else ""
            
            audio_path = Path(audio_file) if Path(audio_file).is_absolute() else dataset_root / audio_file
            if audio_path.exists():
                items.append((idx, str(audio_path), text, instruction))
    
    logger.info(f"Found {len(items)} valid items")
    
    # Load models
    chatterbox, instruction_tokenizer = load_models(args.device)
    t3_config = chatterbox.t3.hp
    
    # Process items
    logger.info("Processing items...")
    cached_items = []
    errors = []
    
    for item in tqdm(items[args.start_idx:], desc="Caching"):
        result, error = process_single_item(
            item, chatterbox, instruction_tokenizer, t3_config
        )
        if result:
            cached_items.append(result)
        else:
            errors.append(error)
        
        # Save batch
        if len(cached_items) >= args.batch_save_size:
            batch_idx = (items.index(item) // args.batch_save_size)
            save_path = output_dir / f"cache_batch_{batch_idx:05d}.pt"
            torch.save(cached_items, save_path)
            logger.info(f"Saved {len(cached_items)} items to {save_path}")
            cached_items = []
    
    # Save remaining
    if cached_items:
        batch_idx = (len(items) // args.batch_save_size)
        save_path = output_dir / f"cache_batch_{batch_idx:05d}.pt"
        torch.save(cached_items, save_path)
        logger.info(f"Saved {len(cached_items)} items to {save_path}")
    
    # Save metadata about cache
    cache_meta = {
        "total_items": len(items) - len(errors),
        "errors": len(errors),
        "batch_size": args.batch_save_size,
        "source_metadata": str(args.metadata_file),
    }
    torch.save(cache_meta, output_dir / "cache_meta.pt")
    
    # Log errors
    if errors:
        logger.warning(f"Encountered {len(errors)} errors:")
        for err in errors[:10]:
            logger.warning(f"  - {err}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more")
    
    logger.info(f"Done! Cache saved to {output_dir}")
    logger.info(f"  - Total items: {len(items) - len(errors)}")
    logger.info(f"  - Errors: {len(errors)}")


if __name__ == "__main__":
    main()
