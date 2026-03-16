#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
OpenTSLM Training Script (Local/Direct GPU)

Run this directly on a machine with GPU (e.g., SageMaker Code Editor VM).
Trains the 5-stage curriculum learning pipeline.

Usage:
    # Full training (all 5 stages)
    python scripts/train_opentslm.py

    # Train specific stages
    python scripts/train_opentslm.py --stages stage1_mcq stage2_captioning

    # Resume from a specific stage
    python scripts/train_opentslm.py --stages stage3_cot stage4_sleep_cot stage5_ecg_cot

    # Quick test (1 epoch per stage)
    python scripts/train_opentslm.py --quick

Estimated time (ml.g5.2xlarge / A10G):
- Stage 1 (TSQA): ~2-3 hours
- Stage 2 (M4): ~3-4 hours
- Stage 3 (HAR): ~4-5 hours
- Stage 4 (Sleep): ~2-3 hours
- Stage 5 (ECG): ~12-15 hours
- Total: ~25 hours
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add original_opentslm to path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
OPENTSLM_DIR = REPO_ROOT / "original_opentslm"


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("GPU: Apple Silicon (MPS)")
            return "mps"
        else:
            print("WARNING: No GPU detected, training will be very slow!")
            return "cpu"
    except ImportError:
        print("PyTorch not installed")
        return "cpu"


def check_hf_token():
    """Check HuggingFace token."""
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        print(f"HuggingFace token: {'*' * 8}{token[-4:]}")
        return True
    else:
        print("WARNING: No HuggingFace token found!")
        print("Set HF_TOKEN environment variable or run: huggingface-cli login")
        return False


def install_dependencies():
    """Install OpenTSLM and dependencies."""
    print("\nInstalling dependencies...")

    # Install opentslm in editable mode
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", str(OPENTSLM_DIR)
    ], check=True)

    # Install additional dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "scikit-learn", "wfdb"
    ], check=True)

    print("Dependencies installed.")


def run_training(
    model: str = "OpenTSLMSP",
    llm_id: str = "meta-llama/Llama-3.2-1B",
    stages: list = None,
    batch_size: int = 4,
    gradient_checkpointing: bool = True,
    device: str = None,
    verbose: bool = False,
):
    """Run curriculum learning training."""

    cmd = [
        sys.executable,
        str(OPENTSLM_DIR / "curriculum_learning.py"),
        "--model", model,
        "--llm_id", llm_id,
        "--batch_size", str(batch_size),
    ]

    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    if device:
        cmd.extend(["--device", device])

    if stages:
        cmd.extend(["--stages"] + stages)

    if verbose:
        cmd.append("--verbose")

    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    # Change to opentslm directory
    os.chdir(OPENTSLM_DIR)

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"Training completed in {elapsed/3600:.1f} hours")
    print(f"Exit code: {result.returncode}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Train OpenTSLM (Local/Direct GPU)')

    # Model options
    parser.add_argument('--model', type=str, default='OpenTSLMSP',
                        choices=['OpenTSLMSP', 'OpenTSLMFlamingo'],
                        help='Model type')
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Base LLM')

    # Training options
    parser.add_argument('--stages', nargs='+',
                        choices=[
                            'stage1_mcq',
                            'stage2_captioning',
                            'stage3_cot',
                            'stage4_sleep_cot',
                            'stage5_ecg_cot',
                        ],
                        default=None,
                        help='Stages to train (default: all)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu)')

    # Convenience options
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (reduces training)')
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                        help='Disable gradient checkpointing')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')
    parser.add_argument('--install', action='store_true',
                        help='Install dependencies and exit')

    args = parser.parse_args()

    print("=" * 60)
    print("OpenTSLM Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LLM: {args.llm}")
    print(f"Stages: {args.stages or 'all'}")
    print(f"Batch size: {args.batch_size}")

    # Check GPU
    device = args.device or check_gpu()

    # Check HuggingFace token
    check_hf_token()

    # Install dependencies
    if args.install:
        install_dependencies()
        return

    # Always ensure dependencies are installed
    install_dependencies()

    # Quick mode - just run stage 1 for testing
    if args.quick:
        print("\nQUICK MODE: Running stage 1 only for testing")
        args.stages = ['stage1_mcq']

    # Run training
    exit_code = run_training(
        model=args.model,
        llm_id=args.llm,
        stages=args.stages,
        batch_size=args.batch_size,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device=device,
        verbose=args.verbose,
    )

    # Print results location
    print("\nResults saved to:")
    results_dir = OPENTSLM_DIR / "results"
    if results_dir.exists():
        for item in results_dir.rglob("best_model.pt"):
            print(f"  {item}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
