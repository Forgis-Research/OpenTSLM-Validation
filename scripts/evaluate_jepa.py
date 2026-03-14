#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Evaluation script for IndustrialJEPA.

Computes comprehensive metrics for JEPA pre-trained models.

Usage:
    python scripts/evaluate_jepa.py --model_path outputs/jepa_full/jepa_run_*/model.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from train_jepa import IndustrialJEPA, CMAPSSDataset


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Extract embeddings from JEPA model."""
    model.eval()
    all_embeddings = []
    all_ruls = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            rul = batch['rul'].numpy()

            output = model(x, compute_loss=False)
            emb = output['embeddings'].mean(dim=1).cpu().numpy()

            all_embeddings.extend(emb)
            all_ruls.extend(rul)

    return np.array(all_embeddings), np.array(all_ruls)


def linear_probe_evaluation(
    train_emb: np.ndarray,
    train_rul: np.ndarray,
    test_emb: np.ndarray,
    test_rul: np.ndarray,
) -> Dict[str, float]:
    """Evaluate representations with a linear probe."""
    # Train linear regressor
    model = Ridge(alpha=1.0)
    model.fit(train_emb, train_rul)

    # Predict
    train_pred = model.predict(train_emb)
    test_pred = model.predict(test_emb)

    # Metrics
    metrics = {
        'train_mae': mean_absolute_error(train_rul, train_pred),
        'train_rmse': np.sqrt(mean_squared_error(train_rul, train_pred)),
        'train_r2': r2_score(train_rul, train_pred),
        'test_mae': mean_absolute_error(test_rul, test_pred),
        'test_rmse': np.sqrt(mean_squared_error(test_rul, test_pred)),
        'test_r2': r2_score(test_rul, test_pred),
    }

    # NASA scoring function
    def nasa_score(pred, true):
        diff = pred - true
        return np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1).sum()

    metrics['test_nasa_score'] = nasa_score(test_pred, test_rul)

    return metrics


def compute_jepa_quality(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute JEPA prediction quality metrics."""
    model.eval()
    total_loss = 0
    total_cosine = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            output = model(x, compute_loss=True)

            total_loss += output['jepa_loss'].item()
            total_cosine += output['cosine_loss'].item()
            n_batches += 1

    return {
        'jepa_loss': total_loss / n_batches,
        'cosine_loss': total_cosine / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate IndustrialJEPA')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--subset', type=str, default='FD001')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    data_path = project_root / 'data' / 'cmapss'
    train_dataset = CMAPSSDataset(data_path, args.subset, 'train', window_size=64)
    test_dataset = CMAPSSDataset(data_path, args.subset, 'val', window_size=64)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Load model
    model = IndustrialJEPA(
        input_channels=train_dataset.num_channels,
        embed_dim=128,
        encoder_layers=4,
        encoder_heads=4,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded: {args.model_path}")

    # Extract features
    print("\nExtracting features...")
    train_emb, train_rul = extract_features(model, train_loader, device)
    test_emb, test_rul = extract_features(model, test_loader, device)

    # Linear probe
    print("Running linear probe evaluation...")
    probe_metrics = linear_probe_evaluation(train_emb, train_rul, test_emb, test_rul)

    # JEPA quality
    print("Computing JEPA quality...")
    jepa_metrics = compute_jepa_quality(model, test_loader, device)

    # Results
    print("\n" + "=" * 60)
    print("INDUSTRIALJEPA EVALUATION RESULTS")
    print("=" * 60)

    print("\nLinear Probe RUL Prediction:")
    print(f"  Train MAE: {probe_metrics['train_mae']:.2f} cycles")
    print(f"  Train RMSE: {probe_metrics['train_rmse']:.2f} cycles")
    print(f"  Test MAE: {probe_metrics['test_mae']:.2f} cycles")
    print(f"  Test RMSE: {probe_metrics['test_rmse']:.2f} cycles")
    print(f"  Test R²: {probe_metrics['test_r2']:.4f}")
    print(f"  NASA Score: {probe_metrics['test_nasa_score']:.2f}")

    print("\nJEPA Pre-training Quality:")
    print(f"  JEPA Loss: {jepa_metrics['jepa_loss']:.4f}")
    print(f"  Cosine Loss: {jepa_metrics['cosine_loss']:.4f}")

    # Save results (convert numpy types to Python types)
    def convert_to_python(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        return obj

    results = {
        'model_path': str(args.model_path),
        'subset': args.subset,
        'probe_metrics': convert_to_python(probe_metrics),
        'jepa_metrics': convert_to_python(jepa_metrics),
    }

    output_path = Path(args.model_path).parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
