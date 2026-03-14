#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Training script for Industrial World Model.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --data_dir ./data --output_dir ./checkpoints
"""

import argparse
import yaml
from pathlib import Path

import torch

from industrialworldlm import IndustrialWorldLM, IndustrialWorldLMConfig
from industrialworldlm.training import IndustrialWorldLMTrainer, TrainingConfig
from industrialworldlm.data import IndustrialDataModule, DatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Industrial World Model")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["cmapss"],
        help="Datasets to train on",
    )

    # Model
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["base", "large", "xl"],
        default="large",
        help="Model size preset",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=24,
        help="Number of input channels",
    )

    # Training
    parser.add_argument(
        "--stage",
        type=str,
        choices=["tokenizer", "dynamics", "full", "finetune"],
        default="full",
        help="Training stage",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="industrial-world-lm",
        help="W&B project name",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )

    # Config file (overrides other args)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    print("=" * 60)
    print("Industrial World Model Training")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Stage: {args.stage}")
    print(f"Datasets: {args.datasets}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create model config
    model_config = IndustrialWorldLMConfig.from_preset(
        args.model_size,
        input_channels=args.input_channels,
    )

    # Create model
    print("\nInitializing model...")
    model = IndustrialWorldLM(model_config, device=args.device)
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")

    # Create data module
    print("\nLoading datasets...")
    dataset_config = DatasetConfig(
        window_size=512,
        stride=64,
        normalize=True,
        augment=(args.stage != "test"),
    )

    data_module = IndustrialDataModule(
        data_dir=args.data_dir,
        datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        config=dataset_config,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create training config
    training_config = TrainingConfig(
        stage=args.stage,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation,
        output_dir=args.output_dir,
        use_amp=args.amp,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        resume_from=args.resume,
    )

    # Create trainer
    trainer = IndustrialWorldLMTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
    )

    # Train!
    print("\nStarting training...")
    trainer.train()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
