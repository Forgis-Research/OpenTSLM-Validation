#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Evaluation script for Industrial World Model.

Usage:
    python scripts/evaluate.py --checkpoint ./checkpoints/best/model.pt --data_dir ./data
    python scripts/evaluate.py --checkpoint ./checkpoints/best/model.pt --dataset cmapss
"""

import argparse
import json
from pathlib import Path

import torch

from industrialworldlm import IndustrialWorldLM
from industrialworldlm.evaluation import (
    run_full_benchmark,
    evaluate_cmapss,
    evaluate_bearing,
    evaluate_tep,
    evaluate_swat,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Industrial World Model")

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cmapss", "bearing", "tep", "swat", "all"],
        help="Dataset to evaluate on (default: all)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Industrial World Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = IndustrialWorldLM.from_pretrained(args.checkpoint, device=args.device)
    model.eval()
    print(f"Model parameters: {model.get_num_params():,}")

    # Run evaluation
    data_dir = Path(args.data_dir)

    if args.dataset is None or args.dataset == "all":
        # Full benchmark
        results = run_full_benchmark(
            model,
            str(data_dir),
            output_path=args.output,
            device=args.device,
        )
    else:
        # Single dataset evaluation
        eval_fns = {
            "cmapss": evaluate_cmapss,
            "bearing": evaluate_bearing,
            "tep": evaluate_tep,
            "swat": evaluate_swat,
        }

        eval_fn = eval_fns[args.dataset]
        result = eval_fn(
            model,
            str(data_dir / args.dataset),
            device=args.device,
            batch_size=args.batch_size,
        )

        print(f"\n{result.dataset} - {result.task}")
        print("-" * 40)
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump({
                    "dataset": result.dataset,
                    "task": result.task,
                    "metrics": result.metrics,
                }, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
