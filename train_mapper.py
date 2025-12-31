"""
Training Script for Instruction Mapper (Stage 1) - FIXED VERSION
=================================================================
This script trains:
- InstructionEncoder (Query + Attention Pooling) - T5 remains frozen
- InstructionMapper (Flow Matching backbone + Output Heads)

To predict:
- Speaker Embedding (256-dim) for T3
- X-Vector (192-dim) for S3

FIXES (v2):
- Removed trainable LatentProjector to avoid moving target / collapse issue
- Target x_1 is now FIXED: direct concatenation of GT embeddings
- Added Reconstruction Loss to train output heads (head_spk, head_xvec)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    print("Warning: wandb not installed. Logging disabled.")

# Chatterbox imports
from chatterbox.models.t3.modules.instruction_encoder import InstructionEncoder
from chatterbox.models.t3.modules.instruction_mapper import InstructionMapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for latent space construction
SPK_EMB_DIM = 256
XVEC_DIM = 192
LATENT_DIM = 512  # SPK_EMB_DIM + XVEC_DIM + padding (64) = 512


def create_fixed_target(spk_emb, x_vector):
    """
    Create FIXED target for Flow Matching by concatenating GT embeddings.
    x_1 = [SpkEmb (256), XVec (192), ZeroPad (64)] = 512-dim
    
    This is NOT trainable - it's the ground truth target.
    """
    batch_size = spk_emb.shape[0]
    device = spk_emb.device
    
    # Zero padding to reach 512-dim
    padding_dim = LATENT_DIM - SPK_EMB_DIM - XVEC_DIM  # 64
    zero_pad = torch.zeros(batch_size, padding_dim, device=device)
    
    # Concatenate: [B, 256] + [B, 192] + [B, 64] = [B, 512]
    x_1 = torch.cat([spk_emb, x_vector, zero_pad], dim=-1)
    return x_1


def extract_from_latent(latent_z):
    """
    Extract SpkEmb and XVec portions from the latent vector (for GT comparison).
    This is the inverse of create_fixed_target (non-learned).
    """
    spk_emb = latent_z[:, :SPK_EMB_DIM]           # [B, 256]
    x_vector = latent_z[:, SPK_EMB_DIM:SPK_EMB_DIM + XVEC_DIM]  # [B, 192]
    return spk_emb, x_vector


class InstructionMapperDataset(Dataset):
    """
    Dataset for Instruction -> (Speaker Embedding, X-Vector) mapping.
    Format: audio_path|text|instruction (separated by |)
    """
    def __init__(
        self, 
        manifest_path: str, 
        tokenizer,
        max_instruction_len: int = 256,
        max_audio_len: int = 16 * 16000,  # 16 seconds at 16kHz
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_instruction_len = max_instruction_len
        self.max_audio_len = max_audio_len
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('|')
                if len(parts) >= 3:
                    audio_path, text, instruction = parts[0], parts[1], parts[2]
                    if os.path.exists(audio_path):
                        self.samples.append({
                            'audio_path': audio_path,
                            'text': text,
                            'instruction': instruction,
                        })
                    else:
                        if line_idx < 10:  # Only warn for first few
                            logger.warning(f"Audio not found: {audio_path}")
                            
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(sample['audio_path'])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Mono
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0)
        
        # Truncate/Pad audio
        if waveform.shape[0] > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]
        
        # Tokenize instruction
        tokens = self.tokenizer(
            sample['instruction'],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_instruction_len,
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'waveform': waveform,
            'instruction': sample['instruction'],
        }


def collate_fn(batch):
    """Custom collate function for variable length audio."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Pad waveforms
    max_len = max(item['waveform'].shape[0] for item in batch)
    waveforms = []
    for item in batch:
        wav = item['waveform']
        if wav.shape[0] < max_len:
            wav = F.pad(wav, (0, max_len - wav.shape[0]))
        waveforms.append(wav)
    waveforms = torch.stack(waveforms)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'waveform': waveforms,
    }


class GroundTruthExtractor(nn.Module):
    """
    Extract ground truth Speaker Embedding and X-Vector from audio.
    Uses pretrained models from Chatterbox.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load Voice Encoder (for Speaker Embedding - 256 dim)
        from chatterbox.models.voice_encoder import VoiceEncoder
        self.voice_encoder = VoiceEncoder()
        self.voice_encoder.to(device)
        self.voice_encoder.eval()
        
        # Load CAM++ (for X-Vector - 192 dim)
        from chatterbox.models.s3gen.s3gen import CAMPPlus
        self.campplus = CAMPPlus()
        self.campplus.to(device)
        self.campplus.eval()
        
        # Freeze all
        for param in self.voice_encoder.parameters():
            param.requires_grad = False
        for param in self.campplus.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, waveform_16k):
        """
        waveform_16k: [Batch, Samples] at 16kHz
        Returns:
            spk_emb: [Batch, 256]
            x_vector: [Batch, 192]
        """
        # VoiceEncoder expects List[np.ndarray] for embeds_from_wavs
        # Convert batch to list of numpy arrays
        wavs_np = [wav.cpu().numpy() for wav in waveform_16k]
        spk_emb = self.voice_encoder.embeds_from_wavs(wavs_np, sample_rate=16000, as_spk=False)
        spk_emb = torch.from_numpy(spk_emb).float().to(self.device)  # [B, 256]
        
        # CAM++ expects [Batch, Samples]
        x_vector = self.campplus.inference(waveform_16k)  # [B, 192]
        
        return spk_emb, x_vector


def train_epoch(
    encoder: InstructionEncoder,
    mapper: InstructionMapper,
    gt_extractor: GroundTruthExtractor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    recon_weight: float = 0.5,
    use_predicted_recon: bool = False,
):
    """
    Train one epoch with:
    - Flow Matching Loss: MSE between predicted and target velocity
    - Reconstruction Loss: MSE between predicted embeddings and GT embeddings
    
    Args:
        use_predicted_recon: If True (Phase 2), use predicted x_1_pred for recon loss.
                             If False (Phase 1/Warmup), use GT x_1 for stable training.
    """
    encoder.train()
    mapper.train()
    
    total_loss = 0.0
    total_flow_loss = 0.0
    total_recon_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        waveform = batch['waveform'].to(device)
        
        # 1. Get GT embeddings from audio (FIXED, not trainable)
        with torch.no_grad():
            gt_spk_emb, gt_x_vector = gt_extractor(waveform)
        
        # 2. Create FIXED target for Flow Matching
        # x_1 = [SpkEmb, XVec, ZeroPad] - This is the GROUND TRUTH
        x_1 = create_fixed_target(gt_spk_emb, gt_x_vector)
        
        # 3. Get instruction embedding
        style_emb = encoder(input_ids, attention_mask)  # [B, 1024]
        
        # 4. Flow Matching: Sample noise and timestep
        batch_size = style_emb.shape[0]
        x_0 = torch.randn_like(x_1)  # Noise - Starting point
        t = torch.rand(batch_size, device=device)  # Timestep
        
        # Interpolate
        t_expand = t.unsqueeze(1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # 5. Predict velocity
        v_pred = mapper(x_t, t, style_emb)
        
        # 6. Target velocity is (x_1 - x_0)
        target_v = x_1 - x_0
        
        # 7. Flow Matching Loss
        flow_loss = F.mse_loss(v_pred, target_v)
        
        # 8. Reconstruction Loss (Train the output heads!)
        # Phase 1 (Warmup): Use GT x_1 - stable training for heads
        # Phase 2 (End-to-End): Use predicted x_1_pred - tighter integration
        if use_predicted_recon:
            # Estimate x_1 from current position and predicted velocity
            x_1_pred = x_t + v_pred * (1 - t_expand)
            pred_spk_emb, pred_x_vector = mapper.project_to_targets(x_1_pred)
        else:
            # Use GT x_1 directly (stable for initial training)
            pred_spk_emb, pred_x_vector = mapper.project_to_targets(x_1)
        recon_loss = F.mse_loss(pred_spk_emb, gt_spk_emb) + F.mse_loss(pred_x_vector, gt_x_vector)
        
        # 9. Total Loss
        loss = flow_loss + recon_weight * recon_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(mapper.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        total_flow_loss += flow_loss.item()
        total_recon_loss += recon_loss.item()
        num_batches += 1
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'flow': f'{flow_loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}'
        })
    
    return total_loss / num_batches, total_flow_loss / num_batches, total_recon_loss / num_batches


def validate(
    encoder: InstructionEncoder,
    mapper: InstructionMapper,
    gt_extractor: GroundTruthExtractor,
    dataloader: DataLoader,
    device: str,
):
    encoder.eval()
    mapper.eval()
    
    total_spk_cos = 0.0
    total_xvec_cos = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            waveform = batch['waveform'].to(device)
            
            # GT
            gt_spk_emb, gt_x_vector = gt_extractor(waveform)
            
            # Predicted (using ODE inference)
            style_emb = encoder(input_ids, attention_mask)
            pred_spk_emb, pred_x_vector = mapper.inference(style_emb, num_steps=20)  # More steps for accuracy
            
            # Cosine similarity
            spk_cos = F.cosine_similarity(pred_spk_emb, gt_spk_emb, dim=-1).mean()
            xvec_cos = F.cosine_similarity(pred_x_vector, gt_x_vector, dim=-1).mean()
            
            total_spk_cos += spk_cos.item() * input_ids.shape[0]
            total_xvec_cos += xvec_cos.item() * input_ids.shape[0]
            num_samples += input_ids.shape[0]
    
    avg_spk_cos = total_spk_cos / num_samples
    avg_xvec_cos = total_xvec_cos / num_samples
    
    return avg_spk_cos, avg_xvec_cos


def main():
    parser = argparse.ArgumentParser(description="Train Instruction Mapper (Fixed Version)")
    parser.add_argument('--manifest', type=str, required=True, help="Path to training manifest")
    parser.add_argument('--val_manifest', type=str, default=None, help="Path to validation manifest")
    parser.add_argument('--output_dir', type=str, default='./checkpoints/mapper', help="Output directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', help="Device")
    parser.add_argument('--save_every', type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument('--num_workers', type=int, default=8, help="DataLoader num_workers")
    parser.add_argument('--compile', action='store_true', help="Use torch.compile for faster training")
    parser.add_argument('--recon_weight', type=float, default=0.5, help="Weight for reconstruction loss")
    parser.add_argument('--use_predicted_recon', action='store_true', 
                        help="Use predicted x_1 for recon loss (Phase 2). Default: use GT x_1 (Phase 1)")
    parser.add_argument('--resume', type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--reset_best_cos", action="store_true", help="Reset best_cos when resuming (useful for Phase 2)")
    
    # WandB argsimprovement for N epochs")
    parser.add_argument('--wandb_project', type=str, default='instruct-tts-mapper', help="WandB project name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="WandB run name (optional)")
    parser.add_argument('--no_wandb', action='store_true', help="Disable WandB logging")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    
    # Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    
    # Dataset
    logger.info("Loading dataset...")
    full_dataset = InstructionMapperDataset(args.manifest, tokenizer)
    
    if args.val_manifest:
        train_dataset = full_dataset
        val_dataset = InstructionMapperDataset(args.val_manifest, tokenizer)
    else:
        # Auto-split 95/5 if no validation set provided
        logger.info("No validation manifest provided. Splitting training data 95/5...")
        val_size = int(len(full_dataset) * 0.05)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Split result: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Models (NO LatentProjector!)
    logger.info("Initializing models...")
    encoder = InstructionEncoder(model_name="google/flan-t5-large").to(device)
    mapper = InstructionMapper(input_dim=1024, internal_dim=512, spk_emb_dim=256, xvec_dim=192, depth=6).to(device)
    gt_extractor = GroundTruthExtractor(device=device)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_cos_from_resume = -1.0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Handle torch.compile checkpoints (keys have _orig_mod. prefix)
        encoder_state = checkpoint['encoder']
        mapper_state = checkpoint['mapper']
        
        # Strip _orig_mod. prefix if present (from torch.compile)
        def strip_prefix(state_dict, prefix="_orig_mod."):
            return {k.replace(prefix, ""): v for k, v in state_dict.items()}
        
        if any(k.startswith("_orig_mod.") for k in encoder_state.keys()):
            encoder_state = strip_prefix(encoder_state)
            logger.info("Stripped _orig_mod. prefix from encoder checkpoint")
        if any(k.startswith("_orig_mod.") for k in mapper_state.keys()):
            mapper_state = strip_prefix(mapper_state)
            logger.info("Stripped _orig_mod. prefix from mapper checkpoint")
        
        encoder.load_state_dict(encoder_state)
        mapper.load_state_dict(mapper_state)
        
        if not args.reset_best_cos:
            if 'best_cos' in checkpoint:
                best_cos_from_resume = checkpoint['best_cos']
                logger.info(f"Resumed from epoch {checkpoint.get('epoch', '?')}, best_cos={best_cos_from_resume:.4f}")
            else:
                logger.info(f"Resumed from epoch {checkpoint.get('epoch', '?')}")
        else:
            logger.info(f"Resumed from epoch {checkpoint.get('epoch', '?')}, but RESETTING best_cos to -1.0")
    
    # Optional: torch.compile for faster training (PyTorch 2.0+)
    if args.compile:
        logger.info("Compiling models with torch.compile...")
        mapper = torch.compile(mapper)
    
    # Optimizer - Only train Query, Attn, Mapper (T5 is frozen, NO projector)
    trainable_params = (
        list(filter(lambda p: p.requires_grad, encoder.parameters())) +
        list(mapper.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Reconstruction Loss Weight: {args.recon_weight}")
    
    # Phase info
    phase_name = "Phase 2 (Predicted Recon)" if args.use_predicted_recon else "Phase 1 (GT Recon)"
    logger.info(f"Training Mode: {phase_name}")
    logger.info(f"Early Stopping Patience: {args.early_stopping_patience} epochs")
    
    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "use_predicted_recon": args.use_predicted_recon,
                "recon_weight": args.recon_weight,
                "early_stopping_patience": args.early_stopping_patience,
                "trainable_params": sum(p.numel() for p in trainable_params),
                "resumed_from": args.resume,
            }
        )
        logger.info(f"WandB initialized: {wandb.run.name}")
    else:
        logger.info("WandB logging disabled.")
    
    # Early Stopping state
    best_cos = best_cos_from_resume  # Use resumed best_cos or -1.0
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, flow_loss, recon_loss = train_epoch(
            encoder, mapper, gt_extractor,
            train_loader, optimizer, device, epoch,
            recon_weight=args.recon_weight,
            use_predicted_recon=args.use_predicted_recon
        )
        scheduler.step()
        
        logger.info(f"Epoch {epoch} [{phase_name}]: Total={train_loss:.4f}, Flow={flow_loss:.4f}, Recon={recon_loss:.4f}")
        
        # WandB logging - Training
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/total_loss": train_loss,
                "train/flow_loss": flow_loss,
                "train/recon_loss": recon_loss,
                "train/phase": 2 if args.use_predicted_recon else 1,
                "learning_rate": scheduler.get_last_lr()[0],
            })
        
        # Validation
        spk_cos, xvec_cos = 0.0, 0.0
        if val_loader is not None:
            spk_cos, xvec_cos = validate(encoder, mapper, gt_extractor, val_loader, device)
            avg_cos = (spk_cos + xvec_cos) / 2
            logger.info(f"Epoch {epoch}: Val SpkCos = {spk_cos:.4f}, XVecCos = {xvec_cos:.4f}")
            
            # WandB logging - Validation
            if use_wandb:
                wandb.log({
                    "val/spk_cos": spk_cos,
                    "val/xvec_cos": xvec_cos,
                    "val/avg_cos": avg_cos,
                })
            
            # Best model saving and early stopping check
            if avg_cos > best_cos:
                best_cos = avg_cos
                epochs_without_improvement = 0
                torch.save({
                    'encoder': encoder.state_dict(),
                    'mapper': mapper.state_dict(),
                    'epoch': epoch,
                    'best_cos': best_cos,
                }, os.path.join(args.output_dir, 'best_model.pt'))
                logger.info(f"Saved best model (cos={avg_cos:.4f})")
                
                if use_wandb:
                    wandb.log({"val/best_cos": best_cos})
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s).")
            
            # Early Stopping
            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {args.early_stopping_patience} epochs)")
                break
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'encoder': encoder.state_dict(),
                'mapper': mapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pt'))
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    # Finish WandB run
    if use_wandb:
        wandb.finish()
    
    logger.info(f"Training complete! Best cos: {best_cos:.4f}")


if __name__ == '__main__':
    main()
