import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import torchaudio
import logging
from tqdm import tqdm
from pathlib import Path

# Load environment variables (for WANDB_API_KEY)
from dotenv import load_dotenv

load_dotenv()

# WandB for experiment tracking
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Chatterbox imports
from chatterbox.models.t3.modules.instruction_encoder import InstructionEncoderT5
from chatterbox.models.t3.modules.instruction_mapper_slice import InstructionMapper

# Constants for latent space construction
SPK_EMB_DIM = 256
XVEC_DIM = 192
LATENT_DIM = SPK_EMB_DIM + XVEC_DIM  # 448 (no padding)


def setup_ddp():
    """Initialize Distributed Data Parallel."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup_ddp():
    """Clean up DDP resources."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return dist.get_rank() == 0


def create_fixed_target(spk_emb, x_vector):
    """
    Create FIXED target for Flow Matching by concatenating GT embeddings.
    x_1 = [SpkEmb (256), XVec (192)] = 448-dim (no padding)
    """
    x_1 = torch.cat([spk_emb, x_vector], dim=-1)
    return x_1


class InstructionMapperDataset(Dataset):
    """Dataset for Instruction -> (Speaker Embedding, X-Vector) mapping."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        voice_encoder,
        campplus,
        max_instruction_len: int = 256,
        max_audio_len: int = 16 * 16000,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_instruction_len = max_instruction_len
        self.max_audio_len = max_audio_len

        # Feature extractors passed from outside (shared across workers)
        self.voice_encoder = voice_encoder
        self.campplus = campplus

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    audio_path, text, instruction = parts[0], parts[1], parts[2]
                    self.samples.append(
                        {
                            "audio_path": audio_path,
                            "text": text,
                            "instruction": instruction,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        waveform, sr = torchaudio.load(sample["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        if waveform.shape[0] > self.max_audio_len:
            waveform = waveform[: self.max_audio_len]

        with torch.no_grad():
            # Voice Encoder (SpkEmb)
            wav_np = waveform.numpy()
            # embeds_from_wavs expects list of arrays
            spk_emb_np = self.voice_encoder.embeds_from_wavs(
                [wav_np], sample_rate=16000, as_spk=False
            )
            spk_emb = torch.from_numpy(spk_emb_np[0]).float()

            # CAMPPlus (X-Vector)
            # campplus expects [Batch, Time] tensor
            wav_tensor = waveform.unsqueeze(0)
            x_vector = self.campplus.inference(wav_tensor)
            x_vector = x_vector.squeeze(0).float()

        tokens = self.tokenizer(
            sample["instruction"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_instruction_len,
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "spk_emb": spk_emb,  # [256]
            "x_vector": x_vector,  # [192]
        }


class CachedInstructionMapperDataset(Dataset):
    """
    Dataset that loads from pre-cached .pt files.
    10-100x faster than InstructionMapperDataset since embeddings are pre-computed.

    Expected cache item keys:
    - speaker_emb: [256] VoiceEncoder embedding
    - x_vector: [192] CAMPPlus embedding
    - instruction_ids: tokenized instruction (already tokenized)
    - instruction: raw instruction text (fallback if instruction_ids missing)
    """

    def __init__(self, cache_dir: str, tokenizer, max_instruction_len: int = 256):
        self.tokenizer = tokenizer
        self.max_instruction_len = max_instruction_len
        self.items = []

        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise ValueError(f"Cache directory not found: {cache_dir}")

        # Load all batch files
        batch_files = sorted(cache_path.glob("cache_batch_*.pt"))
        logging.info(f"Loading cache from {len(batch_files)} batch files...")

        for batch_file in batch_files:
            batch_items = torch.load(batch_file)
            self.items.extend(batch_items)

        logging.info(f"Loaded {len(self.items)} items from cache")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Get embeddings from cache
        spk_emb = item.get("speaker_emb")
        x_vector = item.get("x_vector")

        if spk_emb is None or x_vector is None:
            return None

        # Always tokenize instruction text (more reliable than cached ids)
        instruction_text = item.get("instruction", "")
        tokens = self.tokenizer(
            instruction_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_instruction_len,
        )
        instruction_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return {
            "input_ids": instruction_ids,
            "attention_mask": attention_mask,
            "spk_emb": spk_emb.float(),
            "x_vector": x_vector.float(),
        }


def collate_fn(batch):
    """Simply stack the pre-computed embeddings."""
    # Filter out None samples (failed loads)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None  # DataLoader will skip None batches

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # Stack features directly
    spk_embs = torch.stack([item["spk_emb"] for item in batch])
    x_vectors = torch.stack([item["x_vector"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "gt_spk_emb": spk_embs,
        "gt_x_vector": x_vectors,
    }


def train_epoch(
    encoder_ddp,
    mapper_ddp,
    dataloader,
    optimizer,
    device,
    epoch,
    local_rank=0,
    lambda_spk=1.0,
    lambda_xvec=1.0,
    lambda_cos=0.5,
):
    """Train one epoch with pure Flow Matching."""
    encoder_ddp.train()
    mapper_ddp.train()

    total_loss = 0.0
    num_batches = 0

    mapper_module = mapper_ddp.module if hasattr(mapper_ddp, "module") else mapper_ddp

    # Only show progress bar on main process
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gt_spk_emb = batch["gt_spk_emb"].to(device)
        gt_x_vector = batch["gt_x_vector"].to(device)

        # Create target latent: [SpkEmb | XVec]
        x_1 = create_fixed_target(gt_spk_emb, gt_x_vector)

        # Encode instruction
        style_emb = encoder_ddp(input_ids, attention_mask)

        # Flow Matching: Sample noise and timestep
        batch_size = style_emb.shape[0]
        x_0 = torch.randn_like(x_1)  # Gaussian noise (Standard Normal)
        t = torch.rand(batch_size, device=device)  # Timestep in [0, 1]

        # Interpolation: x_t = (1-t)*x_0 + t*x_1
        t_expand = t.unsqueeze(1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Predict velocity
        v_pred = mapper_ddp(x_t, t, style_emb)

        # Target velocity (straight line from noise to data)
        target_v = x_1 - x_0

        # MSE LOSS
        spk_dim = mapper_module.spk_emb_dim

        v_pred_spk = v_pred[:, :spk_dim]
        v_pred_xvec = v_pred[:, spk_dim:]

        target_v_spk = target_v[:, :spk_dim]
        target_v_xvec = target_v[:, spk_dim:]

        loss_spk = F.mse_loss(v_pred_spk, target_v_spk)
        loss_xvec = F.mse_loss(v_pred_xvec, target_v_xvec)

        # Cosine LOSS
        x1_pred = x_0 + v_pred
        x1_target = x_1

        loss_spk_cos = F.cosine_embedding_loss(
            x1_pred[:, :spk_dim],
            x1_target[:, :spk_dim],
            torch.ones(batch_size, device=device),
        )
        loss_xvec_cos = F.cosine_embedding_loss(
            x1_pred[:, spk_dim:],
            x1_target[:, spk_dim:],
            torch.ones(batch_size, device=device),
        )

        loss_cos = loss_spk_cos + loss_xvec_cos

        # Total LOSS
        loss = lambda_spk * loss_spk + lambda_xvec * loss_xvec + lambda_cos * loss_cos

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(encoder_ddp.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(mapper_ddp.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "l_spk": f"{loss_spk.item():.4f}",
                    "l_xvec": f"{loss_xvec.item():.4f}",
                    "l_cos": f"{loss_cos.item():.4f}",
                }
            )

    return total_loss / num_batches


def validate(
    encoder_ddp,
    mapper_ddp,
    dataloader,
    device,
):
    """Validation loop."""
    encoder_ddp.eval()
    mapper_ddp.eval()

    total_spk_cos = 0.0
    total_xvec_cos = 0.0
    num_samples = 0

    mapper_module = mapper_ddp.module if hasattr(mapper_ddp, "module") else mapper_ddp

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=not is_main_process()):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gt_spk_emb = batch["gt_spk_emb"].to(device)
            gt_x_vector = batch["gt_x_vector"].to(device)

            # Predict via ODE solver
            style_emb = encoder_ddp(input_ids, attention_mask)
            pred_spk_emb, pred_x_vector = mapper_module.inference(style_emb)

            # Cosine similarity
            spk_cos = F.cosine_similarity(pred_spk_emb, gt_spk_emb, dim=-1).mean()
            xvec_cos = F.cosine_similarity(pred_x_vector, gt_x_vector, dim=-1).mean()

            total_spk_cos += spk_cos.item() * input_ids.shape[0]
            total_xvec_cos += xvec_cos.item() * input_ids.shape[0]
            num_samples += input_ids.shape[0]

    # Aggregate across all processes
    metrics = torch.tensor([total_spk_cos, total_xvec_cos, num_samples], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_spk_cos, total_xvec_cos, num_samples = metrics.tolist()

    return total_spk_cos / num_samples, total_xvec_cos / num_samples


def main():
    parser = argparse.ArgumentParser(
        description="Train Instruction Mapper with Pure Flow Matching"
    )
    parser.add_argument(
        "--manifest", type=str, default=None, help="Manifest file (for on-the-fly mode)"
    )
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mapper_slice")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    # Cache arguments (for fast loading from pre-computed embeddings)
    parser.add_argument(
        "--train_cache_dir",
        type=str,
        default=None,
        help="Path to cached training data (from preprocess_cache.py)",
    )
    parser.add_argument(
        "--val_cache_dir", type=str, default=None, help="Path to cached validation data"
    )

    # WandB arguments
    parser.add_argument("--wandb_project", type=str, default="instruct-tts-mapper")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    # Validate arguments
    if not args.train_cache_dir and not args.manifest:
        raise ValueError("Must provide either --train_cache_dir or --manifest")

    # Initialize DDP
    local_rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}"

    # Setup logging (only main process)
    if is_main_process():
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
        logger.info(f"DDP initialized: world_size={world_size}")
        logger.info("PURE FLOW MATCHING - No Scheduled Sampling, No Recon Loss")
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    # ============ Dataset Loading ============
    if args.train_cache_dir:
        # FAST MODE: Load from pre-cached embeddings
        if is_main_process():
            logger.info("=" * 50)
            logger.info("FAST MODE: Loading from cached embeddings")
            logger.info("=" * 50)

        train_dataset = CachedInstructionMapperDataset(args.train_cache_dir, tokenizer)

        if args.val_cache_dir:
            val_dataset = CachedInstructionMapperDataset(args.val_cache_dir, tokenizer)
        else:
            # Split from train cache
            val_size = int(len(train_dataset) * 0.05)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
    else:
        # SLOW MODE: On-the-fly feature extraction using ChatterboxTTS (pretrained weights)
        if is_main_process():
            logger.info("=" * 50)
            logger.info("SLOW MODE: On-the-fly feature extraction")
            logger.info("Using ChatterboxTTS for correct pretrained embeddings")
            logger.info("Consider using --train_cache_dir for 10-100x speedup")
            logger.info("=" * 50)

        # CRITICAL: Must use ChatterboxTTS to get pretrained weights!
        # Standalone VoiceEncoder() and CAMPPlus() do NOT load pretrained weights!
        from chatterbox.tts import ChatterboxTTS

        if is_main_process():
            logger.info("Loading ChatterboxTTS model for embedding extraction...")

        chatterbox = ChatterboxTTS.from_pretrained(device="cpu")  # Load pretrained
        voice_encoder = chatterbox.ve  # Correct VoiceEncoder with weights
        campplus = chatterbox.s3gen.speaker_encoder  # Correct CAMPPlus with weights
        voice_encoder.eval()
        campplus.eval()

        if is_main_process():
            logger.info("Using chatterbox.ve for speaker_emb (256-dim)")
            logger.info("Using chatterbox.s3gen.speaker_encoder for x_vector (192-dim)")

        full_dataset = InstructionMapperDataset(
            args.manifest, tokenizer, voice_encoder, campplus
        )

        if args.val_manifest:
            train_dataset = full_dataset
            val_dataset = InstructionMapperDataset(
                args.val_manifest, tokenizer, voice_encoder, campplus
            )
        else:
            val_size = int(len(full_dataset) * 0.05)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

    if is_main_process():
        logger.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
        )

    # DistributedSampler for training
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Models
    encoder = InstructionEncoderT5(model_name="google/flan-t5-large").to(device)
    mapper = InstructionMapper(
        input_dim=1024, internal_dim=448, spk_emb_dim=256, xvec_dim=192, depth=6
    ).to(device)

    # Resume
    start_epoch = 1
    best_cos = -1.0
    if args.resume:
        if is_main_process():
            logger.info(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)

        # Handle torch.compile checkpoints (keys have _orig_mod. prefix)
        encoder_state = checkpoint["encoder"]
        mapper_state = checkpoint["mapper"]

        # Strip _orig_mod. prefix if present (from torch.compile)
        def strip_prefix(state_dict, prefix="_orig_mod."):
            return {k.replace(prefix, ""): v for k, v in state_dict.items()}

        if any(k.startswith("_orig_mod.") for k in encoder_state.keys()):
            encoder_state = strip_prefix(encoder_state)
            if is_main_process():
                logger.info("Stripped _orig_mod. prefix from encoder checkpoint")
        if any(k.startswith("_orig_mod.") for k in mapper_state.keys()):
            mapper_state = strip_prefix(mapper_state)
            if is_main_process():
                logger.info("Stripped _orig_mod. prefix from mapper checkpoint")

        encoder.load_state_dict(encoder_state)
        mapper.load_state_dict(mapper_state)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_cos" in checkpoint:
            best_cos = checkpoint["best_cos"]

    # Wrap with DDP
    encoder_ddp = DDP(encoder, device_ids=[local_rank])
    mapper_ddp = DDP(mapper, device_ids=[local_rank])

    # Compile (optional)
    if args.compile:
        mapper_ddp = torch.compile(mapper_ddp)
        encoder_ddp = torch.compile(encoder_ddp)

    # Optimizer
    trainable_params = list(
        filter(lambda p: p.requires_grad, encoder_ddp.parameters())
    ) + list(mapper_ddp.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if is_main_process():
        logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
        logger.info(
            f"Batch size per GPU: {args.batch_size}, Total effective batch: {args.batch_size * world_size}"
        )

    # WandB (only main process)
    use_wandb = WANDB_AVAILABLE and not args.no_wandb and is_main_process()
    if use_wandb:
        run_name = args.wandb_run_name or "pure-flow-matching-slice"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
                "learning_rate": args.lr,
                "world_size": world_size,
                "method": "pure_flow_matching",
                "architecture": "slicing",
            },
        )

    # Training loop
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # Important for shuffling

        train_loss = train_epoch(
            encoder_ddp,
            mapper_ddp,
            train_loader,
            optimizer,
            device,
            epoch,
            local_rank=local_rank,
        )
        scheduler.step()

        if is_main_process():
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        # Validation
        spk_cos, xvec_cos = validate(encoder_ddp, mapper_ddp, val_loader, device)
        avg_cos = (spk_cos + xvec_cos) / 2

        # Early stopping flag - must be synchronized across all ranks
        should_stop = torch.tensor([0], device=device)

        if is_main_process():
            logger.info(
                f"Epoch {epoch}: Val SpkCos = {spk_cos:.4f}, XVecCos = {xvec_cos:.4f}, AvgCos = {avg_cos:.4f}"
            )

            if use_wandb:
                wandb.log(
                    {
                        "val/spk_cos": spk_cos,
                        "val/xvec_cos": xvec_cos,
                        "val/avg_cos": avg_cos,
                    }
                )

            if avg_cos > best_cos:
                best_cos = avg_cos
                epochs_without_improvement = 0
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "mapper": mapper.state_dict(),
                        "epoch": epoch,
                        "best_cos": best_cos,
                    },
                    os.path.join(args.output_dir, "best_model.pt"),
                )
                logger.info(f"Saved best model (cos={avg_cos:.4f})")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {epochs_without_improvement} epoch(s)."
                )

            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                should_stop[0] = 1  # Signal to stop

            if epoch % args.save_every == 0:
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "mapper": mapper.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_cos": best_cos,
                    },
                    os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"),
                )

        # Broadcast early stopping decision to all ranks (CRITICAL for DDP sync)
        dist.broadcast(should_stop, src=0)

        # Sync all processes before next epoch or exit
        dist.barrier()

        # All ranks check and break together
        if should_stop[0].item() == 1:
            break

    if use_wandb:
        wandb.finish()

    if is_main_process():
        logger.info(f"Training complete! Best cos: {best_cos:.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
