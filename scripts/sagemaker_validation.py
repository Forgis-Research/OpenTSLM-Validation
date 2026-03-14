#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
SageMaker Validation Launcher for OpenTSLM

Launches validation jobs on AWS SageMaker to reproduce OpenTSLM paper results.

COST ESTIMATE:
- ml.g4dn.xlarge (1x T4 16GB): ~$0.74/hr
- Full validation (4 tasks, ~470K samples): ~2-3 hours
- Total cost: ~$2-3

Prerequisites:
1. AWS CLI configured: aws configure
2. SageMaker execution role
3. HuggingFace token for Llama/Gemma access

Usage:
    # First time setup
    python scripts/sagemaker_validation.py --setup

    # Launch validation job
    python scripts/sagemaker_validation.py --launch

    # Quick validation (100 samples per task)
    python scripts/sagemaker_validation.py --launch --quick

    # Check job status
    python scripts/sagemaker_validation.py --status

    # Download results
    python scripts/sagemaker_validation.py --download <job-name>
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
    'region': 'us-east-1',  # N. Virginia - best GPU availability
    's3_prefix': 'opentslm-validation',

    # Instance recommendations
    'validation_instance': 'ml.g4dn.xlarge',  # $0.74/hr, 1x T4 16GB

    # Cost estimates (per hour, on-demand)
    'instance_costs': {
        'ml.g4dn.xlarge': 0.736,
        'ml.g4dn.2xlarge': 1.12,
        'ml.p3.2xlarge': 3.825,
    }
}


# =============================================================================
# SageMaker Entry Point Script
# =============================================================================

ENTRY_POINT_SCRIPT = '''#!/usr/bin/env python3
"""SageMaker entry point for OpenTSLM validation."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--model", type=str, default="sp")
    parser.add_argument("--llm", type=str, default="llama-3.2-1b")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args, _ = parser.parse_known_args()

    # Set HuggingFace token
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Install opentslm from cloned repo
    print("Installing OpenTSLM...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "/opt/ml/code/original_opentslm"], check=True)

    # HuggingFace login
    if args.hf_token:
        print("Logging into HuggingFace...")
        subprocess.run(["huggingface-cli", "login", "--token", args.hf_token], check=True)

    # Run validation
    cmd = [
        sys.executable, "/opt/ml/code/scripts/validate_opentslm.py",
        "--output-dir", "/opt/ml/output",
        "--model", args.model,
        "--llm", args.llm,
    ]

    if args.task == "all":
        cmd.append("--all")
    else:
        cmd.extend(["--task", args.task])

    if args.quick:
        cmd.append("--quick")
    elif args.max_samples:
        cmd.extend(["--max-samples", str(args.max_samples)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print("Validation complete!")

if __name__ == "__main__":
    main()
'''


# =============================================================================
# Helper Functions
# =============================================================================

def estimate_cost(instance_type: str, quick: bool = False) -> dict:
    """Estimate validation cost."""
    price = CONFIG['instance_costs'].get(instance_type, 1.0)

    if quick:
        hours = 0.5  # ~30 min for quick validation
    else:
        # Full validation times by instance
        hours_map = {
            'ml.g4dn.xlarge': 3.0,
            'ml.g4dn.2xlarge': 2.0,
            'ml.p3.2xlarge': 1.5,
        }
        hours = hours_map.get(instance_type, 3.0)

    return {
        'instance_type': instance_type,
        'estimated_hours': hours,
        'price_per_hour': price,
        'estimated_cost': round(hours * price, 2),
        'quick_mode': quick,
    }


def print_cost_table():
    """Print cost estimates."""
    print("\n" + "="*70)
    print("VALIDATION COST ESTIMATES")
    print("="*70)

    print("\nQuick validation (100 samples/task, ~30 min):")
    print("-"*70)
    for instance in ['ml.g4dn.xlarge', 'ml.g4dn.2xlarge', 'ml.p3.2xlarge']:
        est = estimate_cost(instance, quick=True)
        print(f"  {instance:<20} ${est['price_per_hour']:.2f}/hr  ~${est['estimated_cost']:.2f} total")

    print("\nFull validation (~470K samples, ~2-3 hours):")
    print("-"*70)
    for instance in ['ml.g4dn.xlarge', 'ml.g4dn.2xlarge', 'ml.p3.2xlarge']:
        est = estimate_cost(instance, quick=False)
        print(f"  {instance:<20} ${est['price_per_hour']:.2f}/hr  ~${est['estimated_cost']:.2f} total")

    print("\nRecommendation: ml.g4dn.xlarge for validation (cheapest)")
    print("="*70)


def get_sagemaker_role(session):
    """Get or create SageMaker execution role."""
    iam = session.client('iam')

    # Try to find existing role
    role_name = 'SageMakerExecutionRole'
    try:
        response = iam.get_role(RoleName=role_name)
        return response['Role']['Arn']
    except iam.exceptions.NoSuchEntityException:
        pass

    # Try AmazonSageMaker-ExecutionRole
    try:
        response = iam.get_role(RoleName='AmazonSageMaker-ExecutionRole')
        return response['Role']['Arn']
    except iam.exceptions.NoSuchEntityException:
        pass

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


def launch_validation(
    session,
    instance_type: str = 'ml.g4dn.xlarge',
    task: str = 'all',
    model: str = 'sp',
    llm: str = 'llama-3.2-1b',
    quick: bool = False,
    max_samples: int = None,
    hf_token: str = None,
    spot: bool = False,
):
    """Launch SageMaker validation job."""
    sm_session = sagemaker.Session(boto_session=session)
    bucket = sm_session.default_bucket()
    role = get_sagemaker_role(session)

    # Job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"opentslm-validation-{timestamp}"

    # Estimate cost
    est = estimate_cost(instance_type, quick)

    print(f"\nLaunching validation job: {job_name}")
    print(f"  Instance: {instance_type}")
    print(f"  Task: {task}")
    print(f"  Model: {model} / {llm}")
    print(f"  Quick mode: {quick}")
    print(f"  Spot instances: {spot}")
    print(f"  Estimated cost: ${est['estimated_cost']:.2f}")

    # Get HF token from environment if not provided
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

    if not hf_token:
        print("\nWARNING: No HuggingFace token provided!")
        print("Set HF_TOKEN environment variable or use --hf-token")
        print("Required for Llama/Gemma model access.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return None

    # Create entry point script
    source_dir = Path(__file__).parent.parent
    entry_point_path = source_dir / 'scripts' / 'sagemaker_entry.py'
    entry_point_path.write_text(ENTRY_POINT_SCRIPT)

    # Hyperparameters
    hyperparameters = {
        'task': task,
        'model': model,
        'llm': llm,
    }

    if quick:
        hyperparameters['quick'] = ''  # Flag
    if max_samples:
        hyperparameters['max-samples'] = max_samples
    if hf_token:
        hyperparameters['hf-token'] = hf_token

    # Create estimator
    estimator = PyTorch(
        entry_point='scripts/sagemaker_entry.py',
        source_dir=str(source_dir),
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version='2.1.0',
        py_version='py310',
        sagemaker_session=sm_session,
        hyperparameters=hyperparameters,
        use_spot_instances=spot,
        max_wait=7200 if spot else None,  # 2 hours max wait for spot
        output_path=f"s3://{bucket}/{CONFIG['s3_prefix']}/output",
    )

    # Launch
    estimator.fit(job_name=job_name, wait=False)

    print(f"\nJob submitted: {job_name}")
    print(f"  S3 output: s3://{bucket}/{CONFIG['s3_prefix']}/output/{job_name}")
    print(f"\nCheck status: python scripts/sagemaker_validation.py --status")
    print(f"Download results: python scripts/sagemaker_validation.py --download {job_name}")

    return job_name


def check_status(session):
    """Check status of validation jobs."""
    sm = session.client('sagemaker')

    print("\nRecent OpenTSLM validation jobs:")
    print("-"*70)

    response = sm.list_training_jobs(
        NameContains='opentslm-validation',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=10,
    )

    if not response['TrainingJobSummaries']:
        print("No validation jobs found.")
        return

    for job in response['TrainingJobSummaries']:
        name = job['TrainingJobName']
        status = job['TrainingJobStatus']
        created = job['CreationTime'].strftime('%Y-%m-%d %H:%M')

        details = sm.describe_training_job(TrainingJobName=name)
        instance = details['ResourceConfig']['InstanceType']

        duration = ""
        if 'TrainingEndTime' in details:
            mins = (details['TrainingEndTime'] - details['TrainingStartTime']).total_seconds() / 60
            duration = f"({mins:.0f} min)"

        status_emoji = {
            'Completed': '✅',
            'InProgress': '🔄',
            'Failed': '❌',
            'Stopped': '⏹️',
        }.get(status, '❓')

        print(f"{status_emoji} {name}")
        print(f"   Status: {status} {duration}")
        print(f"   Instance: {instance}")
        print(f"   Created: {created}")
        print()


def download_results(session, job_name: str, local_dir: str = 'outputs/sagemaker'):
    """Download validation results from S3."""
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

    local_file = local_path / 'output.tar.gz'
    s3.download_file(bucket, key, str(local_file))
    print(f"Downloaded to: {local_file}")

    # Extract
    import tarfile
    with tarfile.open(local_file, 'r:gz') as tar:
        tar.extractall(local_path)
    print(f"Extracted to: {local_path}")

    # Print results summary
    for json_file in local_path.glob('*.json'):
        print(f"\nResults from {json_file.name}:")
        with open(json_file) as f:
            data = json.load(f)
            print(json.dumps(data, indent=2)[:1000])


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SageMaker OpenTSLM Validation Launcher')

    # Actions
    parser.add_argument('--setup', action='store_true', help='Check setup')
    parser.add_argument('--launch', action='store_true', help='Launch validation job')
    parser.add_argument('--status', action='store_true', help='Check job status')
    parser.add_argument('--download', type=str, metavar='JOB_NAME', help='Download results')
    parser.add_argument('--costs', action='store_true', help='Show cost estimates')

    # Validation options
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'tsqa', 'har', 'sleep', 'ecg'],
                        help='Task to validate')
    parser.add_argument('--model', type=str, default='sp',
                        choices=['sp', 'flamingo'], help='Model type')
    parser.add_argument('--llm', type=str, default='llama-3.2-1b',
                        help='Base LLM')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation (100 samples/task)')
    parser.add_argument('--max-samples', type=int, help='Max samples per task')

    # AWS options
    parser.add_argument('--instance', type=str, default='ml.g4dn.xlarge',
                        help='Instance type')
    parser.add_argument('--region', type=str, default=CONFIG['region'],
                        help='AWS region')
    parser.add_argument('--spot', action='store_true',
                        help='Use spot instances (60-70%% cheaper)')
    parser.add_argument('--hf-token', type=str, help='HuggingFace token')

    args = parser.parse_args()

    # Show costs
    if args.costs:
        print_cost_table()
        return

    # Check SDK
    if not HAS_SAGEMAKER:
        print("Please install SageMaker SDK: pip install sagemaker boto3")
        sys.exit(1)

    # Create session
    session = boto3.Session(region_name=args.region)

    if args.setup:
        print("Checking SageMaker setup...")
        print(f"  Region: {args.region}")
        try:
            role = get_sagemaker_role(session)
            print(f"  Role: {role}")
        except Exception as e:
            print(f"  Role: ERROR - {e}")

        sm_session = sagemaker.Session(boto_session=session)
        print(f"  Bucket: {sm_session.default_bucket()}")
        print("\nSetup looks good! Ready to launch validation.")

    elif args.launch:
        launch_validation(
            session,
            instance_type=args.instance,
            task=args.task,
            model=args.model,
            llm=args.llm,
            quick=args.quick,
            max_samples=args.max_samples,
            hf_token=args.hf_token,
            spot=args.spot,
        )

    elif args.status:
        check_status(session)

    elif args.download:
        download_results(session, args.download)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
