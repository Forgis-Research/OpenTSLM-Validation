#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
IndustrialJEPA Training Script

Standalone training script for testing the JEPA objective on industrial time series.
Designed to run on CPU for feasibility testing.

Key innovations from EchoJEPA adapted for industrial:
1. JEPA objective: Predict in latent space, not raw sensor values
2. Multi-scale masking: Capture fast and slow dynamics
3. Action conditioning: Enable counterfactual reasoning
4. EMA target encoder: Stable training targets

Usage:
    python scripts/train_jepa.py --epochs 10 --cpu
    python scripts/train_jepa.py --epochs 20 --subset FD001
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# Multi-Scale Patch Embedding
# =============================================================================

class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-scale patch embedding for industrial time series.

    Captures dynamics at multiple temporal scales:
    - Fast: Vibration, control dynamics
    - Slow: Degradation trends, long-term drift
    """

    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 128,
        scales: List[int] = [8, 32],
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

        # Positional encoding
        max_len = 512
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] input time series

        Returns:
            [B, T, D] patch embeddings
        """
        B, C, L = x.shape

        # Get embeddings at each scale
        scale_embeddings = []
        min_patches = L // min(self.scales)

        for conv, scale in zip(self.patch_convs, self.scales):
            emb = conv(x)  # [B, D, n_patches]
            emb = emb.transpose(1, 2)  # [B, n_patches, D]

            # Interpolate to match smallest scale
            if emb.shape[1] != min_patches:
                emb = F.interpolate(
                    emb.transpose(1, 2),
                    size=min_patches,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            scale_embeddings.append(emb)

        # Concatenate and fuse
        multi_scale = torch.cat(scale_embeddings, dim=-1)  # [B, T, D*n_scales]
        fused = self.fusion(multi_scale)  # [B, T, D]

        # Add positional encoding
        T = fused.shape[1]
        if T <= self.pos_embed.shape[1]:
            fused = fused + self.pos_embed[:, :T, :]

        return fused


# =============================================================================
# Transformer Encoder
# =============================================================================

class TransformerEncoder(nn.Module):
    """Lightweight transformer encoder for context processing."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
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
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


# =============================================================================
# JEPA Predictor
# =============================================================================

class JEPAPredictor(nn.Module):
    """
    JEPA predictor network.

    Predicts embeddings for masked positions from visible context.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        predictor_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim

        # Project to predictor dimension
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # Positional encoding
        max_len = 512
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, predictor_dim) * 0.02)

        # Predictor transformer
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=num_layers)

        # Project back to target dimension
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context: [B, T, D] context encoder output
            mask: [B, T] boolean mask (True = predict this position)

        Returns:
            [B, T, D] predicted embeddings
        """
        B, T, D = context.shape

        # Project to predictor dimension
        x = self.input_proj(context)

        # Replace masked positions with mask tokens
        mask_tokens = self.mask_token.expand(B, T, -1)
        x = torch.where(mask.unsqueeze(-1).expand_as(x), mask_tokens, x)

        # Add positional encoding
        if T <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :T, :]

        # Run predictor
        x = self.predictor(x)

        # Project back
        return self.output_proj(x)


# =============================================================================
# IndustrialJEPA Model
# =============================================================================

class IndustrialJEPA(nn.Module):
    """
    Complete IndustrialJEPA Model.

    Architecture:
    1. Multi-scale patch embedding
    2. Context encoder (transformer)
    3. EMA target encoder
    4. JEPA predictor
    """

    def __init__(
        self,
        input_channels: int = 17,
        embed_dim: int = 128,
        encoder_layers: int = 4,
        encoder_heads: int = 4,
        predictor_layers: int = 2,
        predictor_heads: int = 4,
        patch_scales: List[int] = [8, 32],
        mask_ratio: float = 0.6,
        mask_block_size: int = 4,
        ema_momentum: float = 0.996,
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
        )

        # Target encoder (EMA of context encoder)
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

        # Target normalization
        self.target_norm = nn.LayerNorm(embed_dim)

        # RUL prediction head
        self.rul_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

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

    def forward(
        self,
        x: torch.Tensor,
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, C, L] input time series
            compute_loss: Whether to compute JEPA loss

        Returns:
            Dict with 'embeddings', 'loss', etc.
        """
        B = x.shape[0]
        device = x.device

        # 1. Patch embedding
        patches = self.patch_embed(x)  # [B, T, D]
        T = patches.shape[1]

        # 2. Context encoding
        context = self.context_encoder(patches)

        result = {'embeddings': context, 'patches': patches}

        if compute_loss:
            # 3. Create mask
            mask = self.create_mask(T, B, device)

            # 4. Target encoding (EMA, no grad)
            with torch.no_grad():
                target = self.target_encoder(patches)
                target = self.target_norm(target)

            # 5. Predict masked embeddings
            predicted = self.predictor(context, mask)

            # 6. Compute JEPA loss
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

    def predict_rul(self, x: torch.Tensor) -> torch.Tensor:
        """Predict RUL from time series."""
        with torch.no_grad():
            embeddings = self.forward(x, compute_loss=False)['embeddings']
            pooled = embeddings.mean(dim=1)
            return self.rul_head(pooled).squeeze(-1)


# =============================================================================
# C-MAPSS Dataset
# =============================================================================

class CMAPSSDataset(Dataset):
    """C-MAPSS Turbofan Engine Degradation Dataset."""

    def __init__(
        self,
        data_path: str,
        subset: str = 'FD001',
        split: str = 'train',
        window_size: int = 64,
        stride: int = 8,
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

        # Find data file
        data_file = None
        for subdir in ['', '6. Turbofan Engine Degradation Simulation Data Set']:
            candidate = self.data_path / subdir / file_name
            if candidate.exists():
                data_file = candidate
                break

        if data_file is None:
            raise FileNotFoundError(f"Cannot find {file_name} in {self.data_path}")

        # Load data
        raw = np.loadtxt(data_file)

        # Parse columns
        unit_ids = raw[:, 0].astype(int)
        cycles = raw[:, 1].astype(int)
        features = raw[:, 2:]  # [N, 24]

        # Group by unit
        unique_units = np.unique(unit_ids)
        self.sequences = []
        self.ruls = []
        all_features = []

        for unit_id in unique_units:
            mask = unit_ids == unit_id
            unit_features = features[mask]
            unit_cycles = cycles[mask]

            # Compute RUL
            max_cycle = unit_cycles.max()
            rul = max_cycle - unit_cycles
            rul = np.clip(rul, 0, self.max_rul)

            self.sequences.append(unit_features)
            self.ruls.append(rul)
            all_features.append(unit_features)

        # Normalization stats
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

        # Create windows
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

        # Extract and normalize
        window = seq[start:end].copy()
        window = (window - self.mean) / self.std

        # Pad if necessary
        if len(window) < self.window_size:
            padding = np.zeros((self.window_size - len(window), window.shape[1]))
            window = np.concatenate([window, padding], axis=0)

        # Transpose to [C, L]
        window = window.T

        # Get RUL
        window_rul = rul[min(end - 1, len(rul) - 1)]

        return {
            'x': torch.from_numpy(window).float(),
            'rul': torch.tensor(window_rul, dtype=torch.float32),
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_jepa = 0
    total_cosine = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        x = batch['x'].to(device)

        output = model(x, compute_loss=True)
        loss = output['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.update_target_encoder()

        total_loss += loss.item()
        total_jepa += output['jepa_loss'].item()
        total_cosine += output['cosine_loss'].item()
        n_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}")

    return {
        'loss': total_loss / n_batches,
        'jepa_loss': total_jepa / n_batches,
        'cosine_loss': total_cosine / n_batches,
    }


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_rul_error = 0
    n_batches = 0
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            rul_true = batch['rul'].to(device)

            output = model(x, compute_loss=True)
            total_loss += output['loss'].item()

            rul_pred = model.predict_rul(x)
            total_rul_error += torch.abs(rul_pred - rul_true).sum().item()
            n_samples += x.shape[0]
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'rul_mae': total_rul_error / n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Train IndustrialJEPA')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--subset', type=str, default='FD001')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    # Device
    device = torch.device('cpu') if args.cpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data path
    data_path = project_root / 'data' / 'cmapss'

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'jepa_run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading C-MAPSS {args.subset}...")
    train_dataset = CMAPSSDataset(data_path, args.subset, 'train', args.window_size)
    val_dataset = CMAPSSDataset(data_path, args.subset, 'val', args.window_size)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Channels: {train_dataset.num_channels}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    print("\nCreating IndustrialJEPA model...")
    model = IndustrialJEPA(
        input_channels=train_dataset.num_channels,
        embed_dim=args.embed_dim,
        encoder_layers=4,
        encoder_heads=4,
        predictor_layers=2,
        predictor_heads=4,
        patch_scales=[8, 32],
        mask_ratio=0.6,
        mask_block_size=4,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    history = {'train_loss': [], 'val_loss': [], 'val_rul_mae': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val RUL MAE: {val_metrics['rul_mae']:.2f}")

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_rul_mae'].append(val_metrics['rul_mae'])

    # Save results
    torch.save(model.state_dict(), run_dir / 'model.pt')
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Generate figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('JEPA Loss')
        axes[0].set_title('IndustrialJEPA Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(epochs, history['train_loss'], 'b-', label='Train')
        axes[1].semilogy(epochs, history['val_loss'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log)')
        axes[1].set_title('Loss (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, history['val_rul_mae'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('RUL MAE')
        axes[2].set_title('RUL Prediction Error')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(run_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved figure: {run_dir / 'training_curves.png'}")

    except ImportError:
        print("matplotlib not available")

    print(f"\nTraining complete!")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Val RUL MAE: {history['val_rul_mae'][-1]:.2f}")
    print(f"Results saved to: {run_dir}")

    return model, history


if __name__ == '__main__':
    main()
