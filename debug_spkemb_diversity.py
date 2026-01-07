#!/usr/bin/env python3
"""
Debug script for InstructionMapper evaluation.

Two main functionalities:
1. Check DIVERSITY: Are predicted embeddings diverse for different instructions?
2. Check ACCURACY: How close are predictions to ground truth from validation cache?

Usage:
    python debug_spkemb_diversity.py \
        --mapper_ckpt ./checkpoints/mapper_slice_v4/best_model.pt \
        --val_cache_dir ./cache/t3_val \
        --num_samples 100
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.models.t3.modules.instruction_encoder import InstructionEncoderT5
from chatterbox.models.t3.modules.instruction_mapper_slice import InstructionMapper
from transformers import AutoTokenizer


def check_diversity(encoder, mapper, tokenizer, device):
    """Check if mapper produces diverse embeddings for different instructions."""
    print("\n" + "=" * 70)
    print("  PART 1: DIVERSITY CHECK")
    print("=" * 70)

    instructions = [
        "An adult female voice, sounding weak, with a distinct New Zealand accent.",
        "A young girl voice, energetic and lively, with a Canadian accent.",
        "A male voice, deep and dark, with an English accent.",
        "A confident male voice speaking with a Canadian accent.",
        "A mature female voice with an Indian accent.",
        "A young girl voice, sounding vulnerable and soft.",
        "A young boy voice, rugged and feeble.",
        "An elderly man with a raspy, tired voice.",
        "A cheerful young woman with an Australian accent.",
        "A monotone robotic-sounding male voice.",
    ]

    print(f"\nProcessing {len(instructions)} test instructions...")

    spk_embs = []
    x_vectors = []

    with torch.no_grad():
        for instr in instructions:
            tokens = tokenizer(
                instr, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)

            style_emb = encoder(input_ids, attention_mask)
            spk_emb, x_vector = mapper.inference(style_emb, num_steps=20)
            spk_embs.append(spk_emb.cpu())
            x_vectors.append(x_vector.cpu())

    spk_embs = torch.cat(spk_embs, dim=0)  # [N, 256]
    x_vectors = torch.cat(x_vectors, dim=0)  # [N, 192]

    # Calculate pairwise cosine similarities
    n = len(instructions)
    spk_cos_matrix = []
    xvec_cos_matrix = []

    for i in range(n):
        for j in range(i + 1, n):
            spk_cos = F.cosine_similarity(
                spk_embs[i : i + 1], spk_embs[j : j + 1]
            ).item()
            xvec_cos = F.cosine_similarity(
                x_vectors[i : i + 1], x_vectors[j : j + 1]
            ).item()
            spk_cos_matrix.append(spk_cos)
            xvec_cos_matrix.append(xvec_cos)

    avg_spk_cos = sum(spk_cos_matrix) / len(spk_cos_matrix)
    avg_xvec_cos = sum(xvec_cos_matrix) / len(xvec_cos_matrix)
    min_spk_cos = min(spk_cos_matrix)
    max_spk_cos = max(spk_cos_matrix)
    min_xvec_cos = min(xvec_cos_matrix)
    max_xvec_cos = max(xvec_cos_matrix)

    print("\n[SpkEmb Predictions (256-dim)]")
    print(f"  Mean: {spk_embs.mean():.4f}, Std: {spk_embs.std():.4f}")
    print(f"  L2 Norm (avg): {spk_embs.norm(dim=1).mean():.4f}")
    print(
        f"  Pairwise Cosine: avg={avg_spk_cos:.4f}, min={min_spk_cos:.4f}, max={max_spk_cos:.4f}"
    )

    print("\n[X-Vector Predictions (192-dim)]")
    print(f"  Mean: {x_vectors.mean():.4f}, Std: {x_vectors.std():.4f}")
    print(f"  L2 Norm (avg): {x_vectors.norm(dim=1).mean():.4f}")
    print(
        f"  Pairwise Cosine: avg={avg_xvec_cos:.4f}, min={min_xvec_cos:.4f}, max={max_xvec_cos:.4f}"
    )

    # Verdict
    print("\n[Diversity Verdict]")
    if avg_spk_cos > 0.85:
        print(f"  ❌ SpkEmb: TOO SIMILAR (avg cos={avg_spk_cos:.4f} > 0.85)")
        print("     → Mapper may be collapsing to few modes")
    elif avg_spk_cos > 0.7:
        print(f"  ⚠️  SpkEmb: Moderately similar (avg cos={avg_spk_cos:.4f})")
    else:
        print(f"  ✅ SpkEmb: Good diversity (avg cos={avg_spk_cos:.4f})")

    if avg_xvec_cos > 0.85:
        print(f"  ❌ X-Vector: TOO SIMILAR (avg cos={avg_xvec_cos:.4f} > 0.85)")
    elif avg_xvec_cos > 0.7:
        print(f"  ⚠️  X-Vector: Moderately similar (avg cos={avg_xvec_cos:.4f})")
    else:
        print(f"  ✅ X-Vector: Good diversity (avg cos={avg_xvec_cos:.4f})")

    return avg_spk_cos, avg_xvec_cos


def check_accuracy(encoder, mapper, tokenizer, val_cache_dir, device):
    """Check cosine similarity between predictions and ground truth from cache."""
    print("\n" + "=" * 70)
    print("  PART 2: ACCURACY CHECK (vs Ground Truth Cache)")
    print("=" * 70)

    cache_path = Path(val_cache_dir)
    if not cache_path.exists():
        print(f"  ❌ Cache directory not found: {val_cache_dir}")
        return None, None

    # Load cache items
    print(f"\nLoading validation cache from {val_cache_dir}...")
    all_items = []
    for batch_file in sorted(cache_path.glob("cache_batch_*.pt")):
        items = torch.load(batch_file)
        all_items.extend(items)
    print(f"  Loaded {len(all_items)} items")

    # Filter items with valid embeddings
    valid_items = [
        item
        for item in all_items
        if item.get("speaker_emb") is not None
        and item.get("x_vector") is not None
        and item.get("instruction")
    ]
    print(f"  Valid items: {len(valid_items)}")
    print(f"  Evaluating ALL {len(valid_items)} samples...")

    spk_cos_scores = []
    xvec_cos_scores = []

    from tqdm import tqdm

    with torch.no_grad():
        for item in tqdm(valid_items, desc="Checking accuracy"):
            instruction = item["instruction"]
            gt_spk_emb = item["speaker_emb"].to(device)
            gt_x_vector = item["x_vector"].to(device)

            # Tokenize and predict
            tokens = tokenizer(
                instruction, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = tokens.input_ids.to(device)
            attention_mask = tokens.attention_mask.to(device)

            style_emb = encoder(input_ids, attention_mask)
            pred_spk_emb, pred_x_vector = mapper.inference(style_emb, num_steps=20)

            # Cosine similarity
            spk_cos = F.cosine_similarity(pred_spk_emb, gt_spk_emb.unsqueeze(0)).item()
            xvec_cos = F.cosine_similarity(
                pred_x_vector, gt_x_vector.unsqueeze(0)
            ).item()

            spk_cos_scores.append(spk_cos)
            xvec_cos_scores.append(xvec_cos)

    avg_spk_cos = sum(spk_cos_scores) / len(spk_cos_scores)
    avg_xvec_cos = sum(xvec_cos_scores) / len(xvec_cos_scores)
    min_spk_cos = min(spk_cos_scores)
    max_spk_cos = max(spk_cos_scores)
    min_xvec_cos = min(xvec_cos_scores)
    max_xvec_cos = max(xvec_cos_scores)

    print("\n[Prediction vs Ground Truth Cosine Similarity]")
    print(
        f"  SpkEmb:   avg={avg_spk_cos:.4f}, min={min_spk_cos:.4f}, max={max_spk_cos:.4f}"
    )
    print(
        f"  X-Vector: avg={avg_xvec_cos:.4f}, min={min_xvec_cos:.4f}, max={max_xvec_cos:.4f}"
    )

    # Verdict
    print("\n[Accuracy Verdict]")
    if avg_spk_cos < 0.3:
        print(f"  ❌ SpkEmb: Poor accuracy (avg cos={avg_spk_cos:.4f} < 0.3)")
        print("     → Mapper predictions don't match ground truth")
    elif avg_spk_cos < 0.5:
        print(f"  ⚠️  SpkEmb: Low accuracy (avg cos={avg_spk_cos:.4f})")
    else:
        print(f"  ✅ SpkEmb: Reasonable accuracy (avg cos={avg_spk_cos:.4f})")

    if avg_xvec_cos < 0.3:
        print(f"  ❌ X-Vector: Poor accuracy (avg cos={avg_xvec_cos:.4f} < 0.3)")
    elif avg_xvec_cos < 0.5:
        print(f"  ⚠️  X-Vector: Low accuracy (avg cos={avg_xvec_cos:.4f})")
    else:
        print(f"  ✅ X-Vector: Reasonable accuracy (avg cos={avg_xvec_cos:.4f})")

    return avg_spk_cos, avg_xvec_cos


def main():
    parser = argparse.ArgumentParser(description="Debug InstructionMapper outputs")
    parser.add_argument(
        "--mapper_ckpt",
        type=str,
        default="checkpoints/mapper_slice_v4/best_model.pt",
        help="Path to mapper checkpoint",
    )
    parser.add_argument(
        "--val_cache_dir",
        type=str,
        default="./cache/t3_val",
        help="Path to validation cache directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for accuracy check",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  INSTRUCTION MAPPER DEBUG")
    print("=" * 70)
    print(f"\nCheckpoint: {args.mapper_ckpt}")
    print(f"Val Cache:  {args.val_cache_dir}")
    print(f"Device:     {device}")

    # Load models
    print("\nLoading models...")
    ckpt = torch.load(args.mapper_ckpt, map_location="cpu")

    encoder = InstructionEncoderT5(model_name="google/flan-t5-large")
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    encoder.to(device).eval()

    mapper = InstructionMapper()
    mapper.load_state_dict(ckpt["mapper"], strict=False)
    mapper.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    print("  ✓ Encoder and Mapper loaded")

    # Part 1: Diversity Check
    div_spk, div_xvec = check_diversity(encoder, mapper, tokenizer, device)

    # Part 2: Accuracy Check
    acc_spk, acc_xvec = check_accuracy(
        encoder, mapper, tokenizer, args.val_cache_dir, device
    )

    # Final Summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  Diversity (lower = more diverse):")
    print(f"    SpkEmb avg pairwise cos:   {div_spk:.4f}")
    print(f"    X-Vector avg pairwise cos: {div_xvec:.4f}")
    if acc_spk is not None:
        print(f"\n  Accuracy (higher = better match to GT):")
        print(f"    SpkEmb avg cos with GT:    {acc_spk:.4f}")
        print(f"    X-Vector avg cos with GT:  {acc_xvec:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
