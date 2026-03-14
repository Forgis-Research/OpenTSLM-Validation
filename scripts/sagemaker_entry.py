#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
SageMaker entry point for OpenTSLM validation.

This script is executed by SageMaker to run validation.
It installs dependencies, authenticates with HuggingFace, and runs validation.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required packages."""
    print("Installing dependencies...")

    # Install opentslm from cloned repo
    opentslm_dir = Path("/opt/ml/code/original_opentslm")
    if opentslm_dir.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(opentslm_dir)],
            check=True
        )
    else:
        # Install from PyPI
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "opentslm"],
            check=True
        )

    # Install sklearn for metrics
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "scikit-learn"],
        check=True
    )


def authenticate_huggingface(token: str):
    """Login to HuggingFace Hub."""
    if not token:
        print("WARNING: No HuggingFace token provided")
        return

    print("Authenticating with HuggingFace...")
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_TOKEN"] = token

    try:
        subprocess.run(
            ["huggingface-cli", "login", "--token", token],
            check=True,
            capture_output=True
        )
        print("HuggingFace authentication successful")
    except subprocess.CalledProcessError as e:
        print(f"HuggingFace login failed: {e}")
        # Continue anyway - might work with env var


def run_validation(args):
    """Run the validation script."""
    # Build command
    cmd = [
        sys.executable,
        "/opt/ml/code/scripts/validate_opentslm.py",
        "--output-dir", "/opt/ml/output",
        "--model", args.model,
        "--llm", args.llm,
    ]

    if args.task == "all":
        cmd.append("--all")
    else:
        cmd.extend(["--task", args.task])

    if hasattr(args, 'quick') and args.quick:
        cmd.append("--quick")

    if hasattr(args, 'max_samples') and args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])

    print(f"Running validation: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="SageMaker OpenTSLM Validation Entry Point")

    # Validation parameters (passed as hyperparameters)
    parser.add_argument("--task", type=str, default="all",
                        help="Task to validate (all, tsqa, har, sleep, ecg)")
    parser.add_argument("--model", type=str, default="sp",
                        help="Model type (sp, flamingo)")
    parser.add_argument("--llm", type=str, default="llama-3.2-1b",
                        help="Base LLM")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation mode")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples per task")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token")

    # SageMaker passes unknown args
    args, unknown = parser.parse_known_args()

    print("="*60)
    print("OpenTSLM Validation on SageMaker")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"LLM: {args.llm}")
    print(f"Quick mode: {getattr(args, 'quick', False)}")
    print(f"Max samples: {args.max_samples}")
    print("="*60)

    # Step 1: Install dependencies
    install_dependencies()

    # Step 2: Authenticate with HuggingFace
    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    authenticate_huggingface(hf_token)

    # Step 3: Run validation
    exit_code = run_validation(args)

    # Step 4: Copy results to model output (for download)
    output_dir = Path("/opt/ml/output")
    model_dir = Path("/opt/ml/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    for f in output_dir.glob("*.json"):
        print(f"Copying {f.name} to model output")
        (model_dir / f.name).write_text(f.read_text())

    print("\nValidation complete!")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
