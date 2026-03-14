#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
OpenTSLM Validation Script

This script validates pretrained OpenTSLM models against the paper's reported metrics.
Runs on SageMaker (ml.g4dn.xlarge) or locally with CUDA.

Paper reported metrics:
- TSQA: 97.50% accuracy
- HAR-CoT: 65.44% F1
- Sleep-CoT: 69.88% F1
- ECG-QA-CoT: 40.25% F1

Usage:
    # Run all validations
    python scripts/validate_opentslm.py --all

    # Run specific task
    python scripts/validate_opentslm.py --task tsqa

    # Quick validation (subset of samples)
    python scripts/validate_opentslm.py --all --quick

    # Use specific model variant
    python scripts/validate_opentslm.py --task ecg --model flamingo --llm llama-3.2-3b
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Add original_opentslm to path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
OPENTSLM_DIR = REPO_ROOT / "original_opentslm"
sys.path.insert(0, str(OPENTSLM_DIR))

# Paper reported metrics for validation
PAPER_METRICS = {
    "tsqa": {"metric": "accuracy", "value": 97.50, "samples": 4800},
    "m4": {"metric": "bleu", "value": None, "samples": 10000},  # Captioning - no F1
    "har": {"metric": "f1", "value": 65.44, "samples": 8222},
    "sleep": {"metric": "f1", "value": 69.88, "samples": 930},
    "ecg": {"metric": "f1", "value": 40.25, "samples": 41093},
}

# Model naming conventions
MODEL_VARIANTS = {
    "sp": "SoftPrompt",
    "flamingo": "Flamingo",
}

LLM_VARIANTS = {
    "llama-3.2-1b": "llama-3.2-1b",
    "llama-3.2-3b": "llama-3.2-3b",
    "gemma-3-270m": "gemma-3-270m",
    "gemma-3-1b": "gemma-3-1b-pt",
}


def get_repo_id(task: str, model_type: str = "sp", llm: str = "llama-3.2-1b") -> str:
    """Construct HuggingFace repo ID for a model."""
    llm_name = LLM_VARIANTS.get(llm, llm)
    return f"OpenTSLM/{llm_name}-{task}-{model_type}"


def compute_accuracy(predictions: list, gold_answers: list) -> float:
    """Compute accuracy for TSQA-style multiple choice."""
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        # Extract first answer character (a, b, c, etc.)
        pred_clean = pred.strip().lower()[:3]
        gold_clean = gold.strip().lower()[:3]
        if pred_clean == gold_clean:
            correct += 1
    return (correct / len(predictions)) * 100 if predictions else 0


def compute_f1(predictions: list, gold_answers: list, task: str) -> dict:
    """Compute F1 score for classification tasks."""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    # Extract labels from predictions
    pred_labels = []
    gold_labels = []

    for pred, gold in zip(predictions, gold_answers):
        # Task-specific label extraction
        if task == "har":
            # HAR activities
            activities = ["walking", "running", "standing", "sitting", "lying", "cycling", "nordic walking", "ascending stairs"]
            pred_label = extract_label(pred, activities)
            gold_label = extract_label(gold, activities)
        elif task == "sleep":
            # Sleep stages
            stages = ["wake", "n1", "n2", "n3", "rem"]
            pred_label = extract_label(pred, stages)
            gold_label = extract_label(gold, stages)
        elif task == "ecg":
            # ECG - binary or multi-class depending on question
            pred_label = pred.strip().lower()
            gold_label = gold.strip().lower()
        else:
            pred_label = pred.strip().lower()
            gold_label = gold.strip().lower()

        pred_labels.append(pred_label)
        gold_labels.append(gold_label)

    # Compute metrics
    try:
        accuracy = accuracy_score(gold_labels, pred_labels) * 100
        f1 = f1_score(gold_labels, pred_labels, average='macro', zero_division=0) * 100
        precision = precision_score(gold_labels, pred_labels, average='macro', zero_division=0) * 100
        recall = recall_score(gold_labels, pred_labels, average='macro', zero_division=0) * 100
    except Exception as e:
        print(f"Warning: Could not compute sklearn metrics: {e}")
        # Fallback to simple accuracy
        correct = sum(1 for p, g in zip(pred_labels, gold_labels) if p == g)
        accuracy = (correct / len(pred_labels)) * 100 if pred_labels else 0
        f1 = accuracy
        precision = accuracy
        recall = accuracy

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def extract_label(text: str, valid_labels: list) -> str:
    """Extract a label from model output."""
    text_lower = text.lower()
    for label in valid_labels:
        if label in text_lower:
            return label
    return text_lower[:20]  # Fallback


def validate_task(
    task: str,
    model_type: str = "sp",
    llm: str = "llama-3.2-1b",
    max_samples: Optional[int] = None,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> dict:
    """Validate a single task."""
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
    from torch.utils.data import DataLoader

    # Get dataset class
    if task == "tsqa":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset as DatasetClass
    elif task == "m4":
        from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset as DatasetClass
    elif task == "har":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset as DatasetClass
    elif task == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset as DatasetClass
    elif task == "ecg":
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset as DatasetClass
    else:
        raise ValueError(f"Unknown task: {task}")

    repo_id = get_repo_id(task, model_type, llm)
    print(f"\n{'='*60}")
    print(f"Validating: {repo_id}")
    print(f"Task: {task.upper()}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load model
    print(f"\nLoading model from HuggingFace...")
    start_time = time.time()

    # HAR, Sleep, ECG models have LoRA adapters
    enable_lora = task in ["har", "sleep", "ecg"]
    if enable_lora:
        print(f"Enabling LoRA for {task.upper()} task...")

    try:
        model = OpenTSLM.load_pretrained(repo_id, device=device, enable_lora=enable_lora)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return {"error": str(e), "task": task, "repo_id": repo_id}

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")

    # Load dataset
    print(f"\nLoading {task.upper()} test dataset...")
    try:
        test_dataset = DatasetClass("test", EOS_TOKEN=model.get_eos_token())
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return {"error": str(e), "task": task, "repo_id": repo_id}

    total_samples = len(test_dataset)
    if max_samples:
        total_samples = min(max_samples, total_samples)
    print(f"Dataset size: {len(test_dataset)} (evaluating {total_samples})")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    # Run inference
    print(f"\nRunning inference...")
    predictions = []
    gold_answers = []
    inference_times = []

    for i, batch in enumerate(tqdm(test_loader, total=total_samples, desc=f"Validating {task}")):
        if max_samples and i >= max_samples:
            break

        start = time.time()
        try:
            preds = model.generate(batch, max_new_tokens=200)
        except Exception as e:
            print(f"Warning: Inference failed on sample {i}: {e}")
            preds = ["ERROR"]
        inference_times.append(time.time() - start)

        for sample, pred in zip(batch, preds):
            predictions.append(pred)
            gold_answers.append(sample.get("answer", ""))

    # Compute metrics
    print(f"\nComputing metrics...")
    if task == "tsqa":
        accuracy = compute_accuracy(predictions, gold_answers)
        metrics = {"accuracy": accuracy}
    else:
        metrics = compute_f1(predictions, gold_answers, task)

    # Compare to paper
    paper = PAPER_METRICS[task]
    if paper["value"]:
        diff = metrics.get(paper["metric"], 0) - paper["value"]
        status = "PASS" if abs(diff) < 5 else ("CLOSE" if abs(diff) < 10 else "FAIL")
    else:
        diff = None
        status = "N/A"

    results = {
        "task": task,
        "repo_id": repo_id,
        "model_type": model_type,
        "llm": llm,
        "device": device,
        "samples_evaluated": len(predictions),
        "metrics": metrics,
        "paper_metric": paper["metric"],
        "paper_value": paper["value"],
        "difference": diff,
        "status": status,
        "avg_inference_time_ms": sum(inference_times) / len(inference_times) * 1000 if inference_times else 0,
        "total_time_s": sum(inference_times),
        "timestamp": datetime.now().isoformat(),
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {task.upper()}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}%")
    if paper["value"]:
        print(f"\n  Paper reported: {paper['value']:.2f}% {paper['metric']}")
        print(f"  Difference: {diff:+.2f}%")
        print(f"  Status: {status}")
    print(f"\n  Avg inference time: {results['avg_inference_time_ms']:.1f}ms/sample")
    print(f"  Total time: {results['total_time_s']:.1f}s")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"validation_{task}_{model_type}_{llm.replace('-', '_')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate OpenTSLM pretrained models")
    parser.add_argument("--task", type=str, choices=["tsqa", "m4", "har", "sleep", "ecg"],
                        help="Task to validate")
    parser.add_argument("--all", action="store_true", help="Validate all tasks")
    parser.add_argument("--model", type=str, default="sp", choices=["sp", "flamingo"],
                        help="Model type")
    parser.add_argument("--llm", type=str, default="llama-3.2-1b",
                        choices=list(LLM_VARIANTS.keys()),
                        help="Base LLM")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation with subset of samples")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detected if not specified)")
    parser.add_argument("--output-dir", type=str, default="outputs/validation",
                        help="Output directory for results")

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Quick mode limits
    if args.quick:
        args.max_samples = args.max_samples or 100

    output_dir = Path(args.output_dir)

    print("="*60)
    print("OpenTSLM Validation")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Model type: {args.model}")
    print(f"LLM: {args.llm}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Output dir: {output_dir}")

    # Check HuggingFace authentication
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"HuggingFace user: {user.get('name', 'authenticated')}")
    except Exception:
        print("WARNING: Not authenticated with HuggingFace. Run: huggingface-cli login")

    all_results = []

    if args.all:
        tasks = ["tsqa", "har", "sleep", "ecg"]  # Skip M4 for now (captioning metrics different)
    elif args.task:
        tasks = [args.task]
    else:
        parser.print_help()
        return

    for task in tasks:
        try:
            results = validate_task(
                task=task,
                model_type=args.model,
                llm=args.llm,
                max_samples=args.max_samples,
                device=args.device,
                output_dir=output_dir,
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR validating {task}: {e}")
            all_results.append({"task": task, "error": str(e)})

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"\n{'Task':<10} {'Our Result':<15} {'Paper':<15} {'Diff':<10} {'Status'}")
    print("-"*60)

    for r in all_results:
        if "error" in r:
            print(f"{r['task']:<10} ERROR: {r['error'][:30]}")
        else:
            metric = r["paper_metric"]
            our_val = r["metrics"].get(metric, 0)
            paper_val = r.get("paper_value", "N/A")
            diff = r.get("difference", "N/A")
            status = r.get("status", "N/A")

            our_str = f"{our_val:.2f}%" if isinstance(our_val, (int, float)) else str(our_val)
            paper_str = f"{paper_val:.2f}%" if isinstance(paper_val, (int, float)) else str(paper_val)
            diff_str = f"{diff:+.2f}%" if isinstance(diff, (int, float)) else str(diff)

            print(f"{r['task']:<10} {our_str:<15} {paper_str:<15} {diff_str:<10} {status}")

    # Save combined results
    combined_file = output_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_file}")


if __name__ == "__main__":
    main()
