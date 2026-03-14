#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
IndustrialJEPA GPU Training Script - SageMaker Compatible

Optimized for GPU training with:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Multi-GPU support via DDP
- Checkpoint resumption
- SageMaker environment detection
- Weights & Biases logging (optional)

Usage (local):
    python scripts/train_jepa_gpu.py --epochs 100 --batch_size 64

Usage (with wandb):
    python scripts/train_jepa_gpu.py --epochs 100 --wandb --wandb_project industrialjepa

Usage (SageMaker):
    Launched via sagemaker_launcher.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import copy
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast

# Distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Weights & Biases (optional)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SageMaker Environment Detection
# =============================================================================

def is_sagemaker():
    """Check if running in SageMaker."""
    return os.environ.get('SM_TRAINING_ENV') is not None


def get_sagemaker_paths():
    """Get SageMaker input/output paths."""
    if is_sagemaker():
        return {
            'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
            'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'),
            'train_dir': os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'),
        }
    return None


# =============================================================================
# Distributed Training Utilities
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if is_sagemaker():
        # SageMaker sets these environment variables
        world_size = int(os.environ.get('SM_NUM_GPUS', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        logger.info(f"Initialized DDP: rank={local_rank}, world_size={world_size}")

    return local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank):
    """Check if this is the main process."""
    return local_rank == 0


# =============================================================================
# Multi-Scale Patch Embedding
# =============================================================================

class MultiScalePatchEmbedding(nn.Module):
    """Multi-scale patch embedding for industrial time series."""

    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 512,
        scales: List[int] = [4, 8, 16, 32],
    ):
        super().__init__()
        self.scales = scales
        self.embed_dim = embed_dim

        # One conv per scale
        self.patch_convs = nn.ModuleList([
            nn.Conv1d(input_channels, embed_dim, kernel_size=scale, stride=scale)
            for scale in scales
        ])

        # Fusion layer
        self.fusion = nn.Linear(embed_dim * len(scales), embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable positional encoding
        max_len = 1024
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape

        scale_embeddings = []
        min_patches = L // min(self.scales)

        for conv, scale in zip(self.patch_convs, self.scales):
            emb = conv(x)
            emb = emb.transpose(1, 2)

            if emb.shape[1] != min_patches:
                emb = F.interpolate(
                    emb.transpose(1, 2),
                    size=min_patches,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            scale_embeddings.append(emb)

        multi_scale = torch.cat(scale_embeddings, dim=-1)
        fused = self.fusion(multi_scale)
        fused = self.norm(fused)

        T = fused.shape[1]
        if T <= self.pos_embed.shape[1]:
            fused = fused + self.pos_embed[:, :T, :]

        return fused


# =============================================================================
# Transformer Encoder with Flash Attention support
# =============================================================================

class TransformerEncoder(nn.Module):
    """Transformer encoder optimized for GPU."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


# =============================================================================
# JEPA Predictor
# =============================================================================

class JEPAPredictor(nn.Module):
    """JEPA predictor network."""

    def __init__(
        self,
        embed_dim: int = 512,
        predictor_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim

        self.input_proj = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        max_len = 1024
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, predictor_dim) * 0.02)

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = context.shape

        x = self.input_proj(context)
        mask_tokens = self.mask_token.expand(B, T, -1)
        x = torch.where(mask.unsqueeze(-1).expand_as(x), mask_tokens, x)

        if T <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :T, :]

        x = self.predictor(x)
        return self.output_proj(x)


# =============================================================================
# IndustrialJEPA Model (Scaled Up)
# =============================================================================

class IndustrialJEPA(nn.Module):
    """
    IndustrialJEPA Model - GPU optimized, scaled up.

    Model sizes:
    - Small: embed_dim=256, layers=4, heads=4, ~5M params
    - Base: embed_dim=512, layers=6, heads=8, ~25M params
    - Large: embed_dim=768, layers=12, heads=12, ~85M params
    """

    def __init__(
        self,
        input_channels: int = 17,
        embed_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        predictor_layers: int = 3,
        predictor_heads: int = 8,
        patch_scales: List[int] = [4, 8, 16, 32],
        mask_ratio: float = 0.6,
        mask_block_size: int = 4,
        ema_momentum: float = 0.996,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.ema_momentum = ema_momentum

        # Patch embedding
        self.patch_embed = MultiScalePatchEmbedding(
            input_channels=input_channels,
            embed_dim=embed_dim,
            scales=patch_scales,
        )

        # Context encoder
        self.context_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=encoder_heads,
            num_layers=encoder_layers,
            ff_dim=embed_dim * 4,
            dropout=dropout,
        )

        # Target encoder (EMA)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=embed_dim // 2,
            num_heads=predictor_heads,
            num_layers=predictor_layers,
        )

        self.target_norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = self.ema_momentum * param_k.data + (1 - self.ema_momentum) * param_q.data

    def create_mask(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create block masking pattern."""
        mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        num_blocks = max(1, int(seq_len * self.mask_ratio / self.mask_block_size))

        for b in range(batch_size):
            for _ in range(num_blocks):
                if seq_len > self.mask_block_size:
                    start = torch.randint(0, seq_len - self.mask_block_size, (1,)).item()
                    mask[b, start:start + self.mask_block_size] = True

        return mask

    def forward(self, x: torch.Tensor, compute_loss: bool = True) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        device = x.device

        # Patch embedding
        patches = self.patch_embed(x)
        T = patches.shape[1]

        # Context encoding
        context = self.context_encoder(patches)

        result = {'embeddings': context, 'patches': patches}

        if compute_loss:
            mask = self.create_mask(T, B, device)

            with torch.no_grad():
                target = self.target_encoder(patches)
                target = self.target_norm(target)

            predicted = self.predictor(context, mask)

            # JEPA Loss
            mask_expanded = mask.unsqueeze(-1).expand_as(predicted)
            l1_diff = torch.abs(predicted - target)
            masked_diff = l1_diff * mask_expanded.float()

            num_masked = mask.sum().float().clamp(min=1)
            jepa_loss = masked_diff.sum() / (num_masked * self.embed_dim)

            # Cosine similarity regularization
            pred_flat = predicted[mask].view(-1, self.embed_dim)
            target_flat = target[mask].view(-1, self.embed_dim)

            if pred_flat.shape[0] > 0:
                pred_norm = F.normalize(pred_flat, dim=-1)
                target_norm_flat = F.normalize(target_flat, dim=-1)
                cosine_sim = (pred_norm * target_norm_flat).sum(dim=-1).mean()
                cosine_loss = 1 - cosine_sim
            else:
                cosine_loss = torch.tensor(0.0, device=device)

            result['loss'] = jepa_loss + 0.1 * cosine_loss
            result['jepa_loss'] = jepa_loss
            result['cosine_loss'] = cosine_loss
            result['mask'] = mask

        return result


# =============================================================================
# C-MAPSS Dataset (GPU optimized)
# =============================================================================

class CMAPSSDataset(Dataset):
    """C-MAPSS Dataset with GPU-friendly loading."""

    def __init__(
        self,
        data_path: str,
        subset: str = 'FD001',
        split: str = 'train',
        window_size: int = 128,
        stride: int = 16,
        max_rul: int = 125,
    ):
        self.data_path = Path(data_path)
        self.subset = subset
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.max_rul = max_rul

        self._load_data()

    def _load_data(self):
        """Load and preprocess C-MAPSS data."""
        file_name = f'train_{self.subset}.txt' if self.split != 'test' else f'test_{self.subset}.txt'

        data_file = None
        for subdir in ['', '6. Turbofan Engine Degradation Simulation Data Set']:
            candidate = self.data_path / subdir / file_name
            if candidate.exists():
                data_file = candidate
                break

        if data_file is None:
            raise FileNotFoundError(f"Cannot find {file_name} in {self.data_path}")

        raw = np.loadtxt(data_file)

        unit_ids = raw[:, 0].astype(int)
        cycles = raw[:, 1].astype(int)
        features = raw[:, 2:]

        unique_units = np.unique(unit_ids)
        self.sequences = []
        self.ruls = []
        all_features = []

        for unit_id in unique_units:
            mask = unit_ids == unit_id
            unit_features = features[mask]
            unit_cycles = cycles[mask]

            max_cycle = unit_cycles.max()
            rul = max_cycle - unit_cycles
            rul = np.clip(rul, 0, self.max_rul)

            self.sequences.append(unit_features)
            self.ruls.append(rul)
            all_features.append(unit_features)

        all_features = np.concatenate(all_features, axis=0)
        self.mean = all_features.mean(axis=0)
        self.std = all_features.std(axis=0) + 1e-8

        # Drop constant sensors
        keep_cols = [i for i in range(24) if i not in [3, 7, 8, 12, 18, 20, 21]]
        self.sequences = [seq[:, keep_cols] for seq in self.sequences]
        self.mean = self.mean[keep_cols]
        self.std = self.std[keep_cols]
        self.num_channels = len(keep_cols)

        # Split train/val
        if self.split == 'train':
            n = int(len(self.sequences) * 0.8)
            self.sequences = self.sequences[:n]
            self.ruls = self.ruls[:n]
        elif self.split == 'val':
            n = int(len(self.sequences) * 0.8)
            self.sequences = self.sequences[n:]
            self.ruls = self.ruls[n:]

        self._create_windows()

    def _create_windows(self):
        """Create sliding windows."""
        self.windows = []
        for seq_idx, (seq, rul) in enumerate(zip(self.sequences, self.ruls)):
            seq_len = len(seq)
            for start in range(0, max(1, seq_len - self.window_size + 1), self.stride):
                end = min(start + self.window_size, seq_len)
                if end - start >= self.window_size // 2:
                    self.windows.append((seq_idx, start, end))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, start, end = self.windows[idx]
        seq = self.sequences[seq_idx]
        rul = self.ruls[seq_idx]

        window = seq[start:end].copy()
        window = (window - self.mean) / self.std

        if len(window) < self.window_size:
            padding = np.zeros((self.window_size - len(window), window.shape[1]))
            window = np.concatenate([window, padding], axis=0)

        window = window.T

        window_rul = rul[min(end - 1, len(rul) - 1)]

        return {
            'x': torch.from_numpy(window).float(),
            'rul': torch.tensor(window_rul, dtype=torch.float32),
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    accumulation_steps=1,
    use_amp=True,
):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_jepa = 0
    total_cosine = 0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        x = batch['x'].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            output = model(x, compute_loss=True)
            loss = output['loss'] / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA (on unwrapped model if DDP)
            if hasattr(model, 'module'):
                model.module.update_target_encoder()
            else:
                model.update_target_encoder()

        total_loss += output['loss'].item()
        total_jepa += output['jepa_loss'].item()
        total_cosine += output['cosine_loss'].item()
        n_batches += 1

        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}: loss={output['loss'].item():.4f}")

    return {
        'loss': total_loss / n_batches,
        'jepa_loss': total_jepa / n_batches,
        'cosine_loss': total_cosine / n_batches,
    }


def evaluate(model, dataloader, device, use_amp=True):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                output = model(x, compute_loss=True)

            total_loss += output['loss'].item()
            n_batches += 1

    return {'loss': total_loss / n_batches}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, history, path):
    """Save training checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
    }
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['history']


# =============================================================================
# Model Size Configurations
# =============================================================================

MODEL_CONFIGS = {
    'small': {
        'embed_dim': 256,
        'encoder_layers': 4,
        'encoder_heads': 4,
        'predictor_layers': 2,
        'predictor_heads': 4,
    },
    'base': {
        'embed_dim': 512,
        'encoder_layers': 6,
        'encoder_heads': 8,
        'predictor_layers': 3,
        'predictor_heads': 8,
    },
    'large': {
        'embed_dim': 768,
        'encoder_layers': 12,
        'encoder_heads': 12,
        'predictor_layers': 4,
        'predictor_heads': 12,
    },
}


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train IndustrialJEPA on GPU')

    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # Model params
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--mask_ratio', type=float, default=0.6)

    # Data params
    parser.add_argument('--subset', type=str, default='FD001')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)

    # Output params
    parser.add_argument('--output_dir', type=str, default='outputs/jepa_gpu')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)

    # Training options
    # Note: SageMaker passes booleans as strings, so we need to handle that
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--use_amp', type=str2bool, nargs='?', const=True, default=True)
    # Weights & Biases
    parser.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False, help='Enable W&B logging')
    parser.add_argument('--wandb_project', type=str, default='industrialjepa')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    # Setup distributed training
    local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process(local_rank):
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Model size: {args.model_size}")
        logger.info(f"Mixed precision: {args.use_amp}")

    # Data paths
    sm_paths = get_sagemaker_paths()
    if sm_paths:
        data_path = Path(sm_paths['train_dir'])
        output_dir = Path(sm_paths['model_dir'])
    else:
        data_path = Path(args.data_path) if args.data_path else project_root / 'data' / 'cmapss'
        output_dir = Path(args.output_dir)

    # Create output directory
    if is_main_process(local_rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_dir / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = None

    # Sync run_dir across processes
    if world_size > 1:
        run_dir_list = [run_dir]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]

    # Load data
    if is_main_process(local_rank):
        logger.info(f"Loading C-MAPSS {args.subset} from {data_path}...")

    train_dataset = CMAPSSDataset(
        data_path, args.subset, 'train',
        window_size=args.window_size, stride=16
    )
    val_dataset = CMAPSSDataset(
        data_path, args.subset, 'val',
        window_size=args.window_size, stride=16
    )

    if is_main_process(local_rank):
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Channels: {train_dataset.num_channels}")

    # Data loaders
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    config = MODEL_CONFIGS[args.model_size]
    model = IndustrialJEPA(
        input_channels=train_dataset.num_channels,
        embed_dim=config['embed_dim'],
        encoder_layers=config['encoder_layers'],
        encoder_heads=config['encoder_heads'],
        predictor_layers=config['predictor_layers'],
        predictor_heads=config['predictor_heads'],
        mask_ratio=args.mask_ratio,
    ).to(device)

    if is_main_process(local_rank):
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.accumulation_steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Resume from checkpoint
    start_epoch = 1
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    if args.resume and Path(args.resume).exists():
        if is_main_process(local_rank):
            logger.info(f"Resuming from {args.resume}")
        start_epoch, history = load_checkpoint(
            model.module if hasattr(model, 'module') else model,
            optimizer, scheduler, scaler, args.resume, device
        )
        start_epoch += 1

    # Initialize Weights & Biases
    use_wandb = args.wandb and HAS_WANDB and is_main_process(local_rank)
    if use_wandb:
        wandb_config = {
            'model_size': args.model_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'window_size': args.window_size,
            'mask_ratio': args.mask_ratio,
            'num_params': sum(p.numel() for p in (model.module if hasattr(model, 'module') else model).parameters()),
            'dataset': args.subset,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"jepa-{args.model_size}-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=wandb_config,
        )
        wandb.watch(model, log='gradients', log_freq=100)
        logger.info("Weights & Biases initialized")
    elif args.wandb and not HAS_WANDB:
        logger.warning("wandb requested but not installed. Run: pip install wandb")

    # Training loop
    best_val_loss = float('inf')

    if is_main_process(local_rank):
        logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        if is_main_process(local_rank):
            logger.info(f"\nEpoch {epoch}/{args.epochs}")
            logger.info("-" * 50)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.use_amp,
        )
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, use_amp=args.use_amp)

        if is_main_process(local_rank):
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['lr'].append(scheduler.get_last_lr()[0])

            # Log to Weights & Biases
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/jepa_loss': train_metrics['jepa_loss'],
                    'train/cosine_loss': train_metrics['cosine_loss'],
                    'val/loss': val_metrics['loss'],
                    'lr': scheduler.get_last_lr()[0],
                })

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, history,
                    run_dir / 'best_model.pt'
                )

            # Save periodic checkpoint
            if epoch % args.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler, epoch, history,
                    run_dir / f'checkpoint_epoch_{epoch}.pt'
                )

    # Save final model
    if is_main_process(local_rank):
        save_checkpoint(
            model, optimizer, scheduler, scaler, args.epochs, history,
            run_dir / 'final_model.pt'
        )

        # Save history
        with open(run_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save config
        config_dict = {
            'model_size': args.model_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'window_size': args.window_size,
            'mask_ratio': args.mask_ratio,
            'num_params': sum(p.numel() for p in (model.module if hasattr(model, 'module') else model).parameters()),
        }
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"\nTraining complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Results saved to: {run_dir}")

        # Finish wandb
        if use_wandb:
            wandb.finish()

    cleanup_distributed()


if __name__ == '__main__':
    main()
