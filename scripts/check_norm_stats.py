#!/usr/bin/env python3
"""
Check Norm Statistics of Cached Embeddings
==========================================
Calculate Mean, Max, Min of L2 Norms for speaker_emb and x_vector
to diagnose normalization issues.

Usage:
    python scripts/check_norm_stats.py \
        --cache_dir ./cache/t3_train \
        --num_samples 10000
"""

import argparse
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_norm_stats(tensor_list):
    """Calculate statistics from a list of norms."""
    if not tensor_list:
        return None
    
    # Stack to tensor for faster computation
    norms = torch.tensor(tensor_list)
    
    return {
        "mean": norms.mean().item(),
        "std": norms.std().item(),
        "min": norms.min().item(),
        "max": norms.max().item(),
        "count": len(norms)
    }

def print_stats(name, stats):
    if stats is None:
        print(f"\n{name}: NO DATA FOUND")
        return

    print(f"\n{'='*20} {name} {'='*20}")
    print(f"Sample Count: {stats['count']}")
    print(f"Mean Norm:    {stats['mean']:.4f}")
    print(f"Std Dev:      {stats['std']:.4f}")
    print(f"Min Norm:     {stats['min']:.4f}")
    print(f"Max Norm:     {stats['max']:.4f}")
    print("-" * 50)
    
    # Recommendation logic
    if stats['mean'] > 5.0:
        print(f"⚠️  WARNING: {name} has very large norm (~{stats['mean']:.2f}).")
        print("   -> Recommendation: Apply Pre-Normalization or L2 Normalize in code.")
    elif 0.9 <= stats['mean'] <= 1.1 and stats['std'] < 0.1:
        print(f"✅ OK: {name} appears to be already normalized (Unit Vector).")
    else:
        print(f"ℹ️  INFO: {name} has raw distribution.")

def main():
    parser = argparse.ArgumentParser(description="Check embedding norm statistics")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cache directory")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    cache_dir = Path(args.cache_dir)
    
    if not cache_dir.exists():
        print(f"Error: Cache directory {cache_dir} does not exist.")
        return

    # 1. Load Data (Lazy load batch files implies loading full list first is safer for random sampling)
    print(f"Loading cache index from {cache_dir}...")
    all_files = sorted(list(cache_dir.glob("cache_batch_*.pt")))
    
    if not all_files:
        print("No cache files found!")
        return

    # Strategy: Load all items to RAM (assuming 300k fits in RAM easily as list of dicts)
    # If OOM, we can change to file-based sampling, but this is faster.
    all_items = []
    print("Reading cache files...")
    for f in tqdm(all_files, desc="Loading batches"):
        try:
            items = torch.load(f)
            all_items.extend(items)
        except Exception as e:
            print(f"Skipping corrupt file {f}: {e}")

    total_items = len(all_items)
    print(f"Total items found: {total_items}")

    # 2. Sampling
    sample_size = min(args.num_samples, total_items)
    print(f"Sampling {sample_size} items for analysis...")
    samples = random.sample(all_items, sample_size)

    # 3. Compute Norms
    spk_norms = []
    xvec_norms = []

    for item in samples:
        # Speaker Embedding (256 dim)
        if "speaker_emb" in item and item["speaker_emb"] is not None:
            # item["speaker_emb"] is likely a tensor. Calculate L2 norm.
            norm = torch.norm(item["speaker_emb"], p=2).item()
            spk_norms.append(norm)

        # X-Vector (192 dim)
        if "x_vector" in item and item["x_vector"] is not None:
            norm = torch.norm(item["x_vector"], p=2).item()
            xvec_norms.append(norm)

    # 4. Report
    print("\n" + "#" * 60)
    print("NORM STATISTICS REPORT")
    print("#" * 60)

    stats_spk = get_norm_stats(spk_norms)
    print_stats("Speaker Embedding (VoiceEncoder)", stats_spk)

    stats_xvec = get_norm_stats(xvec_norms)
    print_stats("X-Vector (CAMPPlus)", stats_xvec)

    # 5. Comparative Analysis
    if stats_spk and stats_xvec:
        ratio = stats_xvec['mean'] / stats_spk['mean']
        print(f"\n📊 COMPARISON:")
        print(f"Ratio (X-Vec / Spk): {ratio:.2f}")
        if ratio > 2.0 or ratio < 0.5:
            print("❌ IMBALANCE DETECTED: One vector type is significantly larger than the other.")
            print("   -> This causes MSE Loss to focus only on the larger vector.")
            print("   -> ACTION: Normalize both inputs to unit length (norm=1) before training.")

if __name__ == "__main__":
    main()
