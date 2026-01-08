#!/usr/bin/env python3
"""
Validate X-Vector Cache Correctness
====================================
Random sample items from cache and re-compute x_vector & speaker_emb
to verify they match the cached values.

IMPORTANT: Uses ChatterboxTTS model to ensure exact same behavior as preprocess_cache.py

Usage:
    python scripts/validate_cache_embeddings.py \
        --cache_dir ./cache/t3_train \
        --num_samples 500 \
        --debug  # Enable detailed debugging
"""

import argparse
import random
import sys
from pathlib import Path
import logging

import torch
import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use the SAME imports as preprocess_cache.py
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3tokenizer import S3_SR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0).float(), b.unsqueeze(0).float(), dim=1
    ).item()


def debug_single_item(item, chatterbox, device):
    """Detailed debugging for a single item using ChatterboxTTS (same as preprocess_cache.py)."""
    logger.info("=" * 60)
    logger.info("DEBUG: Single item analysis (using ChatterboxTTS)")
    logger.info("=" * 60)

    audio_path = item.get("audio_path", "N/A")
    logger.info(f"Audio path: {audio_path}")
    logger.info(
        f"Audio exists: {Path(audio_path).exists() if audio_path != 'N/A' else False}"
    )
    logger.info(f"Item keys: {list(item.keys())}")

    # Check cached embeddings
    if "speaker_emb" in item:
        cached_spk = item["speaker_emb"]
        logger.info(
            f"\nCached speaker_emb: shape={cached_spk.shape}, dtype={cached_spk.dtype}"
        )
        logger.info(f"  L2 norm: {cached_spk.norm().item():.4f}")
        logger.info(
            f"  Mean: {cached_spk.mean().item():.6f}, Std: {cached_spk.std().item():.6f}"
        )
        logger.info(f"  First 5 values: {cached_spk[:5].tolist()}")

    if "x_vector" in item and item["x_vector"] is not None:
        cached_xvec = item["x_vector"]
        logger.info(
            f"\nCached x_vector: shape={cached_xvec.shape}, dtype={cached_xvec.dtype}"
        )
        logger.info(f"  L2 norm: {cached_xvec.norm().item():.4f}")
        logger.info(
            f"  Mean: {cached_xvec.mean().item():.6f}, Std: {cached_xvec.std().item():.6f}"
        )
        logger.info(f"  First 5 values: {cached_xvec[:5].tolist()}")

    # Load audio EXACTLY like preprocess_cache.py
    if not Path(audio_path).exists():
        logger.error("Audio file not found!")
        return

    # Use librosa (same as preprocess_cache.py)
    wav_16k, _ = librosa.load(audio_path, sr=S3_SR, mono=True)
    logger.info(
        f"\nLoaded audio (librosa, {S3_SR}Hz): length={len(wav_16k)} samples, duration={len(wav_16k) / S3_SR:.2f}s"
    )
    logger.info(
        f"  Audio stats: min={wav_16k.min():.4f}, max={wav_16k.max():.4f}, mean={wav_16k.mean():.6f}"
    )

    # Fresh speaker_emb - EXACTLY like preprocess_cache.py line 72-73
    logger.info("\n--- Computing fresh speaker_emb (chatterbox.ve, as_spk=False) ---")
    speaker_emb_np = chatterbox.ve.embeds_from_wavs(
        [wav_16k], sample_rate=S3_SR, as_spk=False
    )
    fresh_spk = torch.from_numpy(speaker_emb_np[0]).float()

    logger.info(f"Fresh speaker_emb: shape={fresh_spk.shape}")
    logger.info(f"  L2 norm: {fresh_spk.norm().item():.4f}")
    logger.info(
        f"  Mean: {fresh_spk.mean().item():.6f}, Std: {fresh_spk.std().item():.6f}"
    )
    logger.info(f"  First 5 values: {fresh_spk[:5].tolist()}")

    if "speaker_emb" in item:
        cos = cosine_similarity(item["speaker_emb"], fresh_spk)
        logger.info(f"  >>> Cosine similarity (cached vs fresh): {cos:.6f}")

    # Fresh x_vector - using chatterbox.s3gen.speaker_encoder (CAMPPlus)
    logger.info("\n--- Computing fresh x_vector (chatterbox.s3gen.speaker_encoder) ---")
    with torch.no_grad():
        wav_tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
        fresh_xvec = (
            chatterbox.s3gen.speaker_encoder.inference(wav_tensor).cpu().squeeze(0)
        )

    logger.info(f"Fresh x_vector: shape={fresh_xvec.shape}")
    logger.info(f"  L2 norm: {fresh_xvec.norm().item():.4f}")
    logger.info(
        f"  Mean: {fresh_xvec.mean().item():.6f}, Std: {fresh_xvec.std().item():.6f}"
    )
    logger.info(f"  First 5 values: {fresh_xvec[:5].tolist()}")

    if "x_vector" in item and item["x_vector"] is not None:
        cos = cosine_similarity(item["x_vector"], fresh_xvec)
        logger.info(f"  >>> Cosine similarity (cached vs fresh): {cos:.6f}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate cached embeddings")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debugging for first sample",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cache_dir = Path(args.cache_dir)

    # Load all cache items
    logger.info("Loading cache files...")
    all_items = []
    for batch_file in sorted(cache_dir.glob("cache_batch_*.pt")):
        items = torch.load(batch_file)
        all_items.extend(items)
    logger.info(f"Loaded {len(all_items)} total items")

    # Random sample
    sample_size = min(args.num_samples, len(all_items))
    samples = random.sample(all_items, sample_size)
    logger.info(f"Validating {sample_size} random samples...")

    # Check what embeddings exist
    sample_item = samples[0]
    has_spk_emb = "speaker_emb" in sample_item
    has_x_vector = "x_vector" in sample_item
    logger.info(f"Cache contains: speaker_emb={has_spk_emb}, x_vector={has_x_vector}")
    logger.info(f"Cache item keys: {list(sample_item.keys())}")

    # Load ChatterboxTTS (same as preprocess_cache.py)
    logger.info("Loading ChatterboxTTS model (same as preprocess_cache.py)...")
    chatterbox = ChatterboxTTS.from_pretrained(device=args.device)

    # DEBUG: Detailed single item analysis
    if args.debug:
        debug_single_item(samples[0], chatterbox, args.device)
        return

    # Validation metrics
    spk_cos_scores = []
    xvec_cos_scores = []
    spk_errors = []
    xvec_errors = []

    for item in tqdm(samples, desc="Validating"):
        audio_path = item.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            continue

        try:
            # Use librosa (same as preprocess_cache.py)
            wav_16k, _ = librosa.load(audio_path, sr=S3_SR, mono=True)
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            continue

        # Validate speaker_emb (using chatterbox.ve with as_spk=False)
        if has_spk_emb and item.get("speaker_emb") is not None:
            try:
                cached_spk = item["speaker_emb"]
                speaker_emb_np = chatterbox.ve.embeds_from_wavs(
                    [wav_16k], sample_rate=S3_SR, as_spk=False
                )
                fresh_spk = torch.from_numpy(speaker_emb_np[0]).float()
                cos = cosine_similarity(cached_spk, fresh_spk)
                spk_cos_scores.append(cos)
            except Exception as e:
                spk_errors.append(str(e))

        # Validate x_vector (using chatterbox.s3gen.speaker_encoder)
        if has_x_vector and item.get("x_vector") is not None:
            try:
                cached_xvec = item["x_vector"]
                with torch.no_grad():
                    wav_tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(args.device)
                    fresh_xvec = (
                        chatterbox.s3gen.speaker_encoder.inference(wav_tensor)
                        .cpu()
                        .squeeze(0)
                    )
                cos = cosine_similarity(cached_xvec, fresh_xvec)
                xvec_cos_scores.append(cos)
            except Exception as e:
                xvec_errors.append(str(e))

    # Report
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    if spk_cos_scores:
        spk_arr = np.array(spk_cos_scores)
        logger.info(f"Speaker Embedding (VoiceEncoder 256-dim):")
        logger.info(f"  Samples validated: {len(spk_cos_scores)}")
        logger.info(
            f"  Cosine Similarity: mean={spk_arr.mean():.6f}, min={spk_arr.min():.6f}, max={spk_arr.max():.6f}"
        )
        logger.info(f"  Errors: {len(spk_errors)}")
        perfect = (spk_arr > 0.9999).sum()
        logger.info(
            f"  Perfect matches (cos > 0.9999): {perfect}/{len(spk_cos_scores)} ({100 * perfect / len(spk_cos_scores):.1f}%)"
        )

    if xvec_cos_scores:
        xvec_arr = np.array(xvec_cos_scores)
        logger.info(f"\nX-Vector (CAMPPlus 192-dim):")
        logger.info(f"  Samples validated: {len(xvec_cos_scores)}")
        logger.info(
            f"  Cosine Similarity: mean={xvec_arr.mean():.6f}, min={xvec_arr.min():.6f}, max={xvec_arr.max():.6f}"
        )
        logger.info(f"  Errors: {len(xvec_errors)}")
        perfect = (xvec_arr > 0.9999).sum()
        logger.info(
            f"  Perfect matches (cos > 0.9999): {perfect}/{len(xvec_cos_scores)} ({100 * perfect / len(xvec_cos_scores):.1f}%)"
        )

    # Overall verdict
    logger.info("=" * 60)
    all_scores = spk_cos_scores + xvec_cos_scores
    if all_scores:
        min_score = min(all_scores)
        if min_score > 0.999:
            logger.info("✅ PASSED: All cached embeddings are correct!")
        elif min_score > 0.99:
            logger.info(
                "⚠️  WARNING: Minor differences detected (likely floating point)"
            )
        else:
            logger.info("❌ FAILED: Significant mismatches detected!")


if __name__ == "__main__":
    main()
