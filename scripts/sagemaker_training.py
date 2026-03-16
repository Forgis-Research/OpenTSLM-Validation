#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
SageMaker Training Launcher for OpenTSLM

Launches training jobs on AWS SageMaker to reproduce OpenTSLM paper results.
Trains the full 5-stage curriculum learning pipeline.

COST ESTIMATE (Llama 3.2 1B + SP):
- ml.g4dn.xlarge (1x T4 16GB): ~$0.74/hr × 30-40hrs = ~$25-30
- ml.g5.2xlarge (1x A10G 24GB): ~$1.52/hr × 20-25hrs = ~$30-40
- ml.p3.2xlarge (1x V100 16GB): ~$3.83/hr × 15-20hrs = ~$60-80

Prerequisites:
1. AWS CLI configured: aws configure
2. SageMaker execution role with S3 access
3. HuggingFace token for Llama access

Usage:
    # Check setup and cost estimates
    python scripts/sagemaker_training.py --setup

    # Launch full training (all 5 stages)
    python scripts/sagemaker_training.py --launch

    # Launch specific stages only
    python scripts/sagemaker_training.py --launch --stages stage1_mcq stage2_captioning

    # Use spot instances (60-70%% cheaper, but may be interrupted)
    python scripts/sagemaker_training.py --launch --spot

    # Check job status
    python scripts/sagemaker_training.py --status

    # Download trained model
    python scripts/sagemaker_training.py --download <job-name>
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import boto3
    import sagemaker
    from sagemaker.pytorch import PyTorch
    HAS_SAGEMAKER = True
except ImportError:
    HAS_SAGEMAKER = False
    print("SageMaker SDK not installed. Run: pip install sagemaker boto3")


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # AWS Settings
    'region': 'us-east-1',
    's3_prefix': 'opentslm-training',

    # Instance recommendations for training
    'training_instance': 'ml.g5.2xlarge',  # $1.52/hr, 1x A10G 24GB - good balance

    # Cost estimates (per hour, on-demand)
    'instance_costs': {
        'ml.g4dn.xlarge': 0.736,   # T4 16GB - cheapest
        'ml.g4dn.2xlarge': 1.12,   # T4 16GB + more CPU/RAM
        'ml.g5.2xlarge': 1.52,     # A10G 24GB - recommended
        'ml.g5.4xlarge': 2.03,     # A10G 24GB + more CPU/RAM
        'ml.p3.2xlarge': 3.825,    # V100 16GB - fastest
    },

    # Training time estimates (hours) per instance
    'training_hours': {
        'ml.g4dn.xlarge': 40,
        'ml.g4dn.2xlarge': 35,
        'ml.g5.2xlarge': 25,
        'ml.g5.4xlarge': 22,
        'ml.p3.2xlarge': 18,
    },

    # Curriculum stages
    'stages': [
        'stage1_mcq',
        'stage2_captioning',
        'stage3_cot',
        'stage4_sleep_cot',
        'stage5_ecg_cot',
    ],
}


# =============================================================================
# SageMaker Entry Point Script
# =============================================================================

TRAINING_ENTRY_SCRIPT = '''#!/usr/bin/env python3
"""SageMaker entry point for OpenTSLM training."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenTSLMSP")
    parser.add_argument("--llm-id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--stages", type=str, default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    args, _ = parser.parse_known_args()

    # Set HuggingFace token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_TOKEN"] = args.hf_token

    # Install opentslm from cloned repo
    print("Installing OpenTSLM...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/opt/ml/code/original_opentslm"], check=True)

    # Install additional dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn", "wfdb"], check=True)

    # HuggingFace login
    if args.hf_token:
        print("Logging into HuggingFace...")
        subprocess.run(["huggingface-cli", "login", "--token", args.hf_token], check=True)

    # Build training command
    cmd = [
        sys.executable, "/opt/ml/code/original_opentslm/curriculum_learning.py",
        "--model", args.model,
        "--llm_id", args.llm_id,
        "--batch_size", str(args.batch_size),
    ]

    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")

    # Handle stages
    if args.stages != "all":
        stages = args.stages.split(",")
        cmd.extend(["--stages"] + stages)

    print(f"Running training: {' '.join(cmd)}")

    # Change to output directory for results
    os.chdir("/opt/ml/code/original_opentslm")

    result = subprocess.run(cmd)

    # Copy results to model output directory
    print("Copying results to output...")
    results_dir = Path("/opt/ml/code/original_opentslm/results")
    output_dir = Path("/opt/ml/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    if results_dir.exists():
        import shutil
        for item in results_dir.iterdir():
            dest = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    print("Training complete!")
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
'''


# =============================================================================
# Helper Functions
# =============================================================================

def estimate_cost(instance_type: str, stages: list = None, spot: bool = False) -> dict:
    """Estimate training cost."""
    price = CONFIG['instance_costs'].get(instance_type, 1.5)
    hours = CONFIG['training_hours'].get(instance_type, 30)

    # Adjust for partial stages
    if stages and len(stages) < 5:
        # Rough estimate: each stage is about 20% of total time
        stage_weights = {
            'stage1_mcq': 0.10,
            'stage2_captioning': 0.15,
            'stage3_cot': 0.15,
            'stage4_sleep_cot': 0.10,
            'stage5_ecg_cot': 0.50,  # ECG is the largest
        }
        hours = hours * sum(stage_weights.get(s, 0.2) for s in stages)

    # Spot instances are typically 60-70% cheaper
    if spot:
        price *= 0.35  # ~65% discount

    return {
        'instance_type': instance_type,
        'estimated_hours': round(hours, 1),
        'price_per_hour': round(price, 2),
        'estimated_cost': round(hours * price, 2),
        'spot': spot,
        'stages': stages or CONFIG['stages'],
    }


def print_cost_table():
    """Print cost estimates."""
    print("\n" + "="*70)
    print("TRAINING COST ESTIMATES (Llama 3.2 1B + SP, Full 5-Stage Curriculum)")
    print("="*70)

    print("\nOn-Demand instances:")
    print("-"*70)
    print(f"{'Instance':<20} {'GPU':<12} {'$/hr':<10} {'Hours':<10} {'Total':<10}")
    print("-"*70)
    for instance in ['ml.g4dn.xlarge', 'ml.g5.2xlarge', 'ml.p3.2xlarge']:
        est = estimate_cost(instance)
        gpu = {'ml.g4dn.xlarge': 'T4 16GB', 'ml.g5.2xlarge': 'A10G 24GB', 'ml.p3.2xlarge': 'V100 16GB'}[instance]
        print(f"{instance:<20} {gpu:<12} ${est['price_per_hour']:<9.2f} ~{est['estimated_hours']:<9.0f} ~${est['estimated_cost']:.2f}")

    print("\nSpot instances (60-70%% cheaper, may be interrupted):")
    print("-"*70)
    for instance in ['ml.g4dn.xlarge', 'ml.g5.2xlarge', 'ml.p3.2xlarge']:
        est = estimate_cost(instance, spot=True)
        gpu = {'ml.g4dn.xlarge': 'T4 16GB', 'ml.g5.2xlarge': 'A10G 24GB', 'ml.p3.2xlarge': 'V100 16GB'}[instance]
        print(f"{instance:<20} {gpu:<12} ${est['price_per_hour']:<9.2f} ~{est['estimated_hours']:<9.0f} ~${est['estimated_cost']:.2f}")

    print("\nRecommendation: ml.g5.2xlarge (A10G) - best price/performance")
    print("For budget: ml.g4dn.xlarge with spot instances (~$10-15)")
    print("="*70)


def get_sagemaker_role(session):
    """Get or create SageMaker execution role."""
    iam = session.client('iam')

    # Try common role names
    role_names = ['SageMakerExecutionRole', 'AmazonSageMaker-ExecutionRole']

    for role_name in role_names:
        try:
            response = iam.get_role(RoleName=role_name)
            return response['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            continue

    # List all roles and find one with SageMaker
    print("Looking for existing SageMaker role...")
    paginator = iam.get_paginator('list_roles')
    for page in paginator.paginate():
        for role in page['Roles']:
            if 'sagemaker' in role['RoleName'].lower():
                print(f"Found role: {role['RoleName']}")
                return role['Arn']

    raise ValueError(
        "No SageMaker execution role found. Please create one:\n"
        "1. Go to IAM console\n"
        "2. Create role for SageMaker\n"
        "3. Attach AmazonSageMakerFullAccess policy\n"
        "4. Name it: SageMakerExecutionRole"
    )


def launch_training(
    session,
    instance_type: str = 'ml.g5.2xlarge',
    model: str = 'OpenTSLMSP',
    llm_id: str = 'meta-llama/Llama-3.2-1B',
    stages: list = None,
    batch_size: int = 4,
    hf_token: str = None,
    spot: bool = False,
    max_run: int = 172800,  # 48 hours max
):
    """Launch SageMaker training job."""
    sm_session = sagemaker.Session(boto_session=session)
    bucket = sm_session.default_bucket()
    role = get_sagemaker_role(session)

    # Job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"opentslm-train-{timestamp}"

    # Estimate cost
    est = estimate_cost(instance_type, stages, spot)

    print(f"\nLaunching training job: {job_name}")
    print(f"  Instance: {instance_type}")
    print(f"  Model: {model}")
    print(f"  LLM: {llm_id}")
    print(f"  Stages: {', '.join(est['stages'])}")
    print(f"  Batch size: {batch_size}")
    print(f"  Spot instances: {spot}")
    print(f"  Estimated time: ~{est['estimated_hours']} hours")
    print(f"  Estimated cost: ~${est['estimated_cost']:.2f}")

    # Get HF token from environment if not provided
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

    if not hf_token:
        print("\nWARNING: No HuggingFace token provided!")
        print("Set HF_TOKEN environment variable or use --hf-token")
        print("Required for Llama model access.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return None

    # Create entry point script
    source_dir = Path(__file__).parent.parent
    entry_point_path = source_dir / 'scripts' / 'sagemaker_train_entry.py'
    entry_point_path.write_text(TRAINING_ENTRY_SCRIPT)

    # Hyperparameters
    hyperparameters = {
        'model': model,
        'llm-id': llm_id,
        'batch-size': batch_size,
        'gradient-checkpointing': '',  # Flag
    }

    if stages:
        hyperparameters['stages'] = ','.join(stages)
    else:
        hyperparameters['stages'] = 'all'

    if hf_token:
        hyperparameters['hf-token'] = hf_token

    # Create estimator
    estimator = PyTorch(
        entry_point='scripts/sagemaker_train_entry.py',
        source_dir=str(source_dir),
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='2.1.0',
        py_version='py310',
        sagemaker_session=sm_session,
        hyperparameters=hyperparameters,
        use_spot_instances=spot,
        max_wait=max_run + 3600 if spot else None,  # +1hr buffer for spot
        max_run=max_run,
        output_path=f"s3://{bucket}/{CONFIG['s3_prefix']}/output",
        checkpoint_s3_uri=f"s3://{bucket}/{CONFIG['s3_prefix']}/checkpoints/{job_name}",
        checkpoint_local_path='/opt/ml/checkpoints',
    )

    # Launch
    estimator.fit(job_name=job_name, wait=False)

    print(f"\nJob submitted: {job_name}")
    print(f"  S3 output: s3://{bucket}/{CONFIG['s3_prefix']}/output/{job_name}")
    print(f"  S3 checkpoints: s3://{bucket}/{CONFIG['s3_prefix']}/checkpoints/{job_name}")
    print(f"\nCheck status: python scripts/sagemaker_training.py --status")
    print(f"Download model: python scripts/sagemaker_training.py --download {job_name}")

    return job_name


def check_status(session):
    """Check status of training jobs."""
    sm = session.client('sagemaker')

    print("\nRecent OpenTSLM training jobs:")
    print("-"*70)

    response = sm.list_training_jobs(
        NameContains='opentslm-train',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10,
    )

    if not response['TrainingJobSummaries']:
        print("No training jobs found.")
        return

    for job in response['TrainingJobSummaries']:
        name = job['TrainingJobName']
        status = job['TrainingJobStatus']
        created = job['CreationTime'].strftime('%Y-%m-%d %H:%M')

        details = sm.describe_training_job(TrainingJobName=name)
        instance = details['ResourceConfig']['InstanceType']

        duration = ""
        if 'TrainingEndTime' in details:
            hours = (details['TrainingEndTime'] - details['TrainingStartTime']).total_seconds() / 3600
            duration = f"({hours:.1f} hrs)"
        elif status == 'InProgress' and 'TrainingStartTime' in details:
            hours = (datetime.now(details['TrainingStartTime'].tzinfo) - details['TrainingStartTime']).total_seconds() / 3600
            duration = f"(running {hours:.1f} hrs)"

        status_emoji = {
            'Completed': '[OK]',
            'InProgress': '[..]',
            'Failed': '[X]',
            'Stopped': '[--]',
        }.get(status, '[?]')

        print(f"{status_emoji} {name}")
        print(f"   Status: {status} {duration}")
        print(f"   Instance: {instance}")
        print(f"   Created: {created}")

        if status == 'Failed':
            print(f"   Reason: {details.get('FailureReason', 'Unknown')[:80]}")
        print()


def download_model(session, job_name: str, local_dir: str = 'outputs/trained_models'):
    """Download trained model from S3."""
    sm = session.client('sagemaker')
    s3 = session.client('s3')

    # Get job details
    try:
        job = sm.describe_training_job(TrainingJobName=job_name)
    except Exception as e:
        print(f"Job not found: {job_name}")
        return

    if job['TrainingJobStatus'] != 'Completed':
        print(f"Job not completed (status: {job['TrainingJobStatus']})")
        if job['TrainingJobStatus'] == 'Failed':
            print(f"Failure reason: {job.get('FailureReason', 'Unknown')}")
        return

    # Get output location
    output_uri = job['ModelArtifacts']['S3ModelArtifacts']
    print(f"Downloading from: {output_uri}")

    # Parse S3 URI
    parts = output_uri.replace('s3://', '').split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''

    # Download
    local_path = Path(local_dir) / job_name
    local_path.mkdir(parents=True, exist_ok=True)

    local_file = local_path / 'model.tar.gz'
    s3.download_file(bucket, key, str(local_file))
    print(f"Downloaded to: {local_file}")

    # Extract
    import tarfile
    with tarfile.open(local_file, 'r:gz') as tar:
        tar.extractall(local_path)
    print(f"Extracted to: {local_path}")

    # List contents
    print("\nModel contents:")
    for item in local_path.rglob('*'):
        if item.is_file():
            rel_path = item.relative_to(local_path)
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.1f} MB)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SageMaker OpenTSLM Training Launcher')

    # Actions
    parser.add_argument('--setup', action='store_true', help='Check setup and show costs')
    parser.add_argument('--launch', action='store_true', help='Launch training job')
    parser.add_argument('--status', action='store_true', help='Check job status')
    parser.add_argument('--download', type=str, metavar='JOB_NAME', help='Download trained model')
    parser.add_argument('--costs', action='store_true', help='Show cost estimates')

    # Training options
    parser.add_argument('--model', type=str, default='OpenTSLMSP',
                        choices=['OpenTSLMSP', 'OpenTSLMFlamingo'],
                        help='Model type')
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Base LLM')
    parser.add_argument('--stages', nargs='+',
                        choices=CONFIG['stages'],
                        default=None,
                        help='Stages to train (default: all)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')

    # AWS options
    parser.add_argument('--instance', type=str, default='ml.g5.2xlarge',
                        help='Instance type')
    parser.add_argument('--region', type=str, default=CONFIG['region'],
                        help='AWS region')
    parser.add_argument('--spot', action='store_true',
                        help='Use spot instances (60-70%% cheaper)')
    parser.add_argument('--hf-token', type=str, help='HuggingFace token')
    parser.add_argument('--max-run', type=int, default=172800,
                        help='Max runtime in seconds (default: 48 hours)')

    args = parser.parse_args()

    # Show costs
    if args.costs or args.setup:
        print_cost_table()
        if not args.setup:
            return

    # Check SDK
    if not HAS_SAGEMAKER:
        print("Please install SageMaker SDK: pip install sagemaker boto3")
        sys.exit(1)

    # Create session
    session = boto3.Session(region_name=args.region)

    if args.setup:
        print("\nChecking SageMaker setup...")
        print(f"  Region: {args.region}")
        try:
            role = get_sagemaker_role(session)
            print(f"  Role: {role}")
        except Exception as e:
            print(f"  Role: ERROR - {e}")

        sm_session = sagemaker.Session(boto_session=session)
        print(f"  Bucket: {sm_session.default_bucket()}")
        print("\nSetup looks good! Ready to launch training.")

    elif args.launch:
        launch_training(
            session,
            instance_type=args.instance,
            model=args.model,
            llm_id=args.llm,
            stages=args.stages,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
            spot=args.spot,
            max_run=args.max_run,
        )

    elif args.status:
        check_status(session)

    elif args.download:
        download_model(session, args.download)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
