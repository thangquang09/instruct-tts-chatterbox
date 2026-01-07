#!/usr/bin/env python3
"""
Append X-Vector to Existing Cache Files
========================================
This script adds x_vector (CAMPPlus 192-dim) to existing cached .pt files
without re-running the full preprocessing pipeline.

IMPORTANT: Uses ChatterboxTTS's s3gen.speaker_encoder to ensure consistency.

Usage:
    python scripts/append_x_vector_cache.py \
        --cache_dir ./cache/t3_train \
        --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path
import logging

import torch
import librosa
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use ChatterboxTTS to get the correct speaker_encoder
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3tokenizer import S3_SR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Append x_vector to existing cache files"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Directory containing cache_batch_*.pt files",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device for inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (larger = faster but more VRAM)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original cache files before modifying",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-compute x_vector even if already exists",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return

    # Find all batch files
    batch_files = sorted(cache_dir.glob("cache_batch_*.pt"))
    if not batch_files:
        logger.error(f"No cache_batch_*.pt files found in {cache_dir}")
        return

    logger.info(f"Found {len(batch_files)} cache batch files")

    # Load ChatterboxTTS to get the correct speaker_encoder
    logger.info(f"Loading ChatterboxTTS model on {args.device}...")
    chatterbox = ChatterboxTTS.from_pretrained(device=args.device)
    speaker_encoder = chatterbox.s3gen.speaker_encoder
    speaker_encoder.eval()
    logger.info("Using chatterbox.s3gen.speaker_encoder for x_vector extraction")

    total_items = 0
    total_updated = 0
    total_skipped = 0
    total_errors = 0

    for batch_file in tqdm(batch_files, desc="Processing batch files"):
        logger.info(f"Processing {batch_file.name}...")

        # Backup if requested
        if args.backup:
            backup_path = batch_file.with_suffix(".pt.bak")
            if not backup_path.exists():
                import shutil

                shutil.copy2(batch_file, backup_path)
                logger.info(f"  Backed up to {backup_path.name}")

        # Load batch
        try:
            items = torch.load(batch_file)
        except Exception as e:
            logger.error(f"  Failed to load {batch_file}: {e}")
            continue

        if not isinstance(items, list):
            logger.warning(f"  Unexpected format in {batch_file}, skipping")
            continue

        # Check if already has x_vector
        sample_item = items[0] if items else {}
        if "x_vector" in sample_item and not args.force:
            logger.info(
                f"  {batch_file.name} already has x_vector, skipping (use --force to recompute)"
            )
            total_skipped += len(items)
            continue

        # Process each item
        batch_errors = 0
        for item in items:
            audio_path = item.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                logger.warning(
                    f"  Item {item.get('idx', '?')}: audio_path missing or not found"
                )
                item["x_vector"] = None
                batch_errors += 1
                continue

            try:
                # Load audio with librosa (same as preprocess_cache.py)
                wav_16k, _ = librosa.load(audio_path, sr=S3_SR, mono=True)

                # Compute x_vector using chatterbox's speaker_encoder
                with torch.no_grad():
                    wav_tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(args.device)
                    x_vector = speaker_encoder.inference(wav_tensor).cpu().squeeze(0)

                item["x_vector"] = x_vector

            except Exception as e:
                logger.warning(f"  Item {item.get('idx', '?')}: Failed: {e}")
                item["x_vector"] = None
                batch_errors += 1

        # Save updated batch
        try:
            torch.save(items, batch_file)
            logger.info(f"  Updated {len(items)} items, {batch_errors} errors")
            total_updated += len(items) - batch_errors
            total_errors += batch_errors
        except Exception as e:
            logger.error(f"  Failed to save {batch_file}: {e}")

        total_items += len(items)

    # Update cache metadata
    meta_path = cache_dir / "cache_meta.pt"
    if meta_path.exists():
        try:
            cache_meta = torch.load(meta_path)
            cache_meta["x_vector_added"] = True
            cache_meta["x_vector_updated"] = total_updated
            cache_meta["x_vector_errors"] = total_errors
            cache_meta["x_vector_source"] = "chatterbox.s3gen.speaker_encoder"
            torch.save(cache_meta, meta_path)
            logger.info("Updated cache_meta.pt")
        except Exception as e:
            logger.warning(f"Failed to update cache_meta.pt: {e}")

    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Total items processed: {total_items}")
    logger.info(f"  Successfully updated:  {total_updated}")
    logger.info(f"  Skipped (had x_vector): {total_skipped}")
    logger.info(f"  Errors:                {total_errors}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
