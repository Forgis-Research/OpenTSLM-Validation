#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Amazon SageMaker Training Launcher for IndustrialJEPA

This script launches training jobs on AWS SageMaker.

Prerequisites:
1. AWS CLI configured with credentials
2. SageMaker execution role with S3 access
3. Data uploaded to S3

Usage:
    # First time setup
    python scripts/sagemaker_launcher.py --setup

    # Upload data to S3
    python scripts/sagemaker_launcher.py --upload-data

    # Launch training
    python scripts/sagemaker_launcher.py --launch --instance-type ml.g4dn.xlarge

    # Launch larger training
    python scripts/sagemaker_launcher.py --launch --instance-type ml.p3.2xlarge --model-size large

    # Check job status
    python scripts/sagemaker_launcher.py --status
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

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

# Default configuration - UPDATE THESE FOR YOUR ACCOUNT
CONFIG = {
    # AWS Settings
    'region': 'eu-central-1',  # Frankfurt
    's3_bucket': None,      # Will be auto-created if None
    's3_prefix': 'industrialjepa',

    # SageMaker Settings
    'role_name': 'SageMakerExecutionRole',  # Your SageMaker execution role

    # Training defaults
    'default_instance': 'ml.g4dn.xlarge',
    'default_epochs': 100,
    'default_batch_size': 64,
}


# =============================================================================
# Instance Types and Pricing (us-east-1, on-demand, per hour)
# =============================================================================

INSTANCE_INFO = {
    # CPU instances
    'ml.m5.xlarge': {
        'gpu': 'None (CPU)',
        'vcpu': 4,
        'memory': '16 GB',
        'price_per_hour': 0.23,
        'recommended_batch': 32,
        'recommended_for': 'Testing and debugging',
    },
    'ml.m5.2xlarge': {
        'gpu': 'None (CPU)',
        'vcpu': 8,
        'memory': '32 GB',
        'price_per_hour': 0.46,
        'recommended_batch': 64,
        'recommended_for': 'Larger CPU training',
    },
    'ml.c5.xlarge': {
        'gpu': 'None (CPU)',
        'vcpu': 4,
        'memory': '8 GB',
        'price_per_hour': 0.204,
        'recommended_batch': 32,
        'recommended_for': 'Compute-optimized CPU',
    },
    # GPU instances
    'ml.g4dn.xlarge': {
        'gpu': '1x T4 (16GB)',
        'vcpu': 4,
        'memory': '16 GB',
        'price_per_hour': 0.736,
        'recommended_batch': 64,
        'recommended_for': 'Development and small experiments',
    },
    'ml.g4dn.2xlarge': {
        'gpu': '1x T4 (16GB)',
        'vcpu': 8,
        'memory': '32 GB',
        'price_per_hour': 1.12,
        'recommended_batch': 128,
        'recommended_for': 'Small to medium training',
    },
    'ml.g4dn.4xlarge': {
        'gpu': '1x T4 (16GB)',
        'vcpu': 16,
        'memory': '64 GB',
        'price_per_hour': 1.686,
        'recommended_batch': 128,
        'recommended_for': 'Medium training with more CPU',
    },
    'ml.g4dn.12xlarge': {
        'gpu': '4x T4 (64GB total)',
        'vcpu': 48,
        'memory': '192 GB',
        'price_per_hour': 5.672,
        'recommended_batch': 256,
        'recommended_for': 'Multi-GPU training',
    },
    'ml.p3.2xlarge': {
        'gpu': '1x V100 (16GB)',
        'vcpu': 8,
        'memory': '61 GB',
        'price_per_hour': 3.825,
        'recommended_batch': 128,
        'recommended_for': 'Fast training, good value',
    },
    'ml.p3.8xlarge': {
        'gpu': '4x V100 (64GB total)',
        'vcpu': 32,
        'memory': '244 GB',
        'price_per_hour': 14.688,
        'recommended_batch': 512,
        'recommended_for': 'Large-scale training',
    },
    'ml.p3.16xlarge': {
        'gpu': '8x V100 (128GB total)',
        'vcpu': 64,
        'memory': '488 GB',
        'price_per_hour': 28.152,
        'recommended_batch': 1024,
        'recommended_for': 'Very large-scale training',
    },
    'ml.p4d.24xlarge': {
        'gpu': '8x A100 (320GB total)',
        'vcpu': 96,
        'memory': '1152 GB',
        'price_per_hour': 37.688,
        'recommended_batch': 2048,
        'recommended_for': 'State-of-the-art training',
    },
}


# =============================================================================
# Cost Estimation
# =============================================================================

def estimate_training_cost(
    instance_type: str,
    model_size: str,
    epochs: int,
    dataset_size: str = 'cmapss',
) -> dict:
    """
    Estimate training costs based on instance and model size.

    Returns estimated hours and cost range.
    """
    if instance_type not in INSTANCE_INFO:
        raise ValueError(f"Unknown instance type: {instance_type}")

    info = INSTANCE_INFO[instance_type]
    price_per_hour = info['price_per_hour']

    # Estimate training time based on model size and GPU
    # These are rough estimates based on similar workloads
    base_hours = {
        'small': {'ml.g4dn.xlarge': 2, 'ml.g4dn.2xlarge': 1.5, 'ml.p3.2xlarge': 1},
        'base': {'ml.g4dn.xlarge': 6, 'ml.g4dn.2xlarge': 4, 'ml.p3.2xlarge': 2.5},
        'large': {'ml.g4dn.xlarge': 15, 'ml.g4dn.2xlarge': 10, 'ml.p3.2xlarge': 6},
    }

    # Get base hours for 100 epochs on C-MAPSS
    if instance_type in base_hours.get(model_size, {}):
        hours_100_epochs = base_hours[model_size][instance_type]
    else:
        # Estimate based on compute power relative to g4dn.xlarge
        g4dn_hours = base_hours[model_size].get('ml.g4dn.xlarge', 5)
        relative_speed = info['price_per_hour'] / INSTANCE_INFO['ml.g4dn.xlarge']['price_per_hour']
        hours_100_epochs = g4dn_hours / (relative_speed ** 0.7)  # Not linear scaling

    # Scale by epochs
    estimated_hours = hours_100_epochs * (epochs / 100)

    # Add overhead (instance startup, data loading, checkpointing)
    overhead_hours = 0.25

    total_hours = estimated_hours + overhead_hours
    estimated_cost = total_hours * price_per_hour

    return {
        'instance_type': instance_type,
        'model_size': model_size,
        'epochs': epochs,
        'estimated_hours': round(total_hours, 2),
        'price_per_hour': price_per_hour,
        'estimated_cost': round(estimated_cost, 2),
        'cost_range': (round(estimated_cost * 0.7, 2), round(estimated_cost * 1.5, 2)),
        'gpu_info': info['gpu'],
    }


def print_cost_table():
    """Print cost estimates for common configurations."""
    print("\n" + "=" * 80)
    print("TRAINING COST ESTIMATES")
    print("=" * 80)
    print("\nEstimated costs for 100 epochs on C-MAPSS FD001:")
    print("-" * 80)
    print(f"{'Instance':<20} {'GPU':<20} {'Model':<8} {'Hours':<8} {'Cost':<12}")
    print("-" * 80)

    for instance in ['ml.g4dn.xlarge', 'ml.g4dn.2xlarge', 'ml.p3.2xlarge']:
        for model_size in ['small', 'base', 'large']:
            est = estimate_training_cost(instance, model_size, 100)
            print(f"{instance:<20} {est['gpu_info']:<20} {model_size:<8} "
                  f"{est['estimated_hours']:<8.1f} ${est['estimated_cost']:<10.2f}")

    print("-" * 80)
    print("\nNotes:")
    print("- Prices are US East (N. Virginia) on-demand rates")
    print("- Actual costs may vary by 30-50% based on data and convergence")
    print("- Multi-GPU instances (p3.8xlarge+) are only cost-effective for large models")
    print("- Consider Spot instances for 60-70% savings (with interruption risk)")


# =============================================================================
# SageMaker Operations
# =============================================================================

def get_or_create_bucket(session, bucket_name=None):
    """Get or create S3 bucket for SageMaker."""
    if bucket_name:
        return bucket_name

    # Use SageMaker default bucket
    sm_session = sagemaker.Session(boto_session=session)
    bucket = sm_session.default_bucket()
    print(f"Using S3 bucket: {bucket}")
    return bucket


def upload_data_to_s3(session, bucket, local_path, s3_prefix):
    """Upload training data to S3."""
    s3 = session.client('s3')

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Data path not found: {local_path}")

    print(f"Uploading {local_path} to s3://{bucket}/{s3_prefix}/")

    for file_path in local_path.glob('**/*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/data/{relative_path}"
            print(f"  Uploading: {relative_path}")
            s3.upload_file(str(file_path), bucket, s3_key)

    print("Data upload complete!")
    return f"s3://{bucket}/{s3_prefix}/data"


def launch_training_job(
    session,
    bucket,
    s3_prefix,
    role_arn,
    instance_type,
    model_size='base',
    epochs=100,
    batch_size=64,
    spot_instances=False,
    wandb_api_key=None,
    wandb_project='industrialjepa',
):
    """Launch SageMaker training job."""
    sm_session = sagemaker.Session(boto_session=session)

    # Job name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_name = f"industrialjepa-{model_size}-{timestamp}"

    print(f"\nLaunching training job: {job_name}")
    print(f"  Instance: {instance_type}")
    print(f"  Model size: {model_size}")
    print(f"  Epochs: {epochs}")
    if wandb_api_key:
        print(f"  W&B logging: enabled (project: {wandb_project})")

    # Estimate cost
    est = estimate_training_cost(instance_type, model_size, epochs)
    print(f"  Estimated cost: ${est['cost_range'][0]:.2f} - ${est['cost_range'][1]:.2f}")

    # Get script directory
    source_dir = str(Path(__file__).parent)

    # Hyperparameters
    hyperparameters = {
        'epochs': epochs,
        'batch_size': batch_size,
        'model_size': model_size,
        'lr': 1e-4,
        'save_every': 10,
    }

    # Add wandb params if API key provided
    if wandb_api_key:
        hyperparameters['wandb'] = True
        hyperparameters['wandb_project'] = wandb_project
        hyperparameters['wandb_run_name'] = job_name

    # Environment variables
    environment = {}
    if wandb_api_key:
        environment['WANDB_API_KEY'] = wandb_api_key

    # Disable AMP for CPU instances (use_amp is True by default, set to False for CPU)
    if 'ml.m' in instance_type or 'ml.c' in instance_type:
        hyperparameters['use_amp'] = False
        print("  Note: Mixed precision disabled for CPU instance")

    # Create estimator
    estimator = PyTorch(
        entry_point='train_jepa_gpu.py',
        source_dir=source_dir,
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        framework_version='2.1.0',
        py_version='py310',
        sagemaker_session=sm_session,
        hyperparameters=hyperparameters,
        environment=environment,
        use_spot_instances=spot_instances,
        max_wait=86400 if spot_instances else None,  # 24 hours max wait for spot
        checkpoint_s3_uri=f"s3://{bucket}/{s3_prefix}/checkpoints/{job_name}",
    )

    # Data location
    s3_data = f"s3://{bucket}/{s3_prefix}/data"

    # Launch training
    estimator.fit(
        inputs={'training': s3_data},
        job_name=job_name,
        wait=False,  # Don't wait for completion
    )

    print(f"\nTraining job submitted!")
    print(f"  Job name: {job_name}")
    print(f"  Check status: python scripts/sagemaker_launcher.py --status")
    print(f"  SageMaker console: https://{session.region_name}.console.aws.amazon.com/sagemaker/home")

    return job_name


def check_job_status(session):
    """Check status of recent training jobs."""
    sm = session.client('sagemaker')

    print("\nRecent IndustrialJEPA training jobs:")
    print("-" * 80)

    response = sm.list_training_jobs(
        NameContains='industrialjepa',
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
        created = job['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')

        # Get more details
        details = sm.describe_training_job(TrainingJobName=name)
        instance_type = details['ResourceConfig']['InstanceType']

        duration = ""
        if 'TrainingEndTime' in details:
            end = details['TrainingEndTime']
            start = details['TrainingStartTime']
            mins = (end - start).total_seconds() / 60
            duration = f"{mins:.1f} min"

        print(f"{name}")
        print(f"  Status: {status}")
        print(f"  Instance: {instance_type}")
        print(f"  Created: {created}")
        if duration:
            print(f"  Duration: {duration}")
        print()


def download_model(session, bucket, job_name, local_path):
    """Download trained model from S3."""
    s3 = session.client('s3')

    # Find model artifacts
    sm = session.client('sagemaker')
    job = sm.describe_training_job(TrainingJobName=job_name)

    if job['TrainingJobStatus'] != 'Completed':
        print(f"Job {job_name} is not completed (status: {job['TrainingJobStatus']})")
        return

    model_uri = job['ModelArtifacts']['S3ModelArtifacts']
    print(f"Downloading model from: {model_uri}")

    # Parse S3 URI
    parts = model_uri.replace('s3://', '').split('/')
    bucket = parts[0]
    key = '/'.join(parts[1:])

    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    local_file = local_path / 'model.tar.gz'
    s3.download_file(bucket, key, str(local_file))
    print(f"Downloaded to: {local_file}")

    # Extract
    import tarfile
    with tarfile.open(local_file, 'r:gz') as tar:
        tar.extractall(local_path)
    print(f"Extracted to: {local_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SageMaker Training Launcher')

    # Actions
    parser.add_argument('--setup', action='store_true', help='Initial setup')
    parser.add_argument('--upload-data', action='store_true', help='Upload data to S3')
    parser.add_argument('--launch', action='store_true', help='Launch training job')
    parser.add_argument('--status', action='store_true', help='Check job status')
    parser.add_argument('--download', type=str, help='Download model from job name')
    parser.add_argument('--costs', action='store_true', help='Show cost estimates')

    # Training options
    parser.add_argument('--instance-type', type=str, default=CONFIG['default_instance'])
    parser.add_argument('--model-size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--epochs', type=int, default=CONFIG['default_epochs'])
    parser.add_argument('--batch-size', type=int, default=CONFIG['default_batch_size'])
    parser.add_argument('--spot', action='store_true', help='Use spot instances')

    # Weights & Biases
    parser.add_argument('--wandb-api-key', type=str, default=None, help='W&B API key')
    parser.add_argument('--wandb-project', type=str, default='industrialjepa', help='W&B project name')

    # AWS options
    parser.add_argument('--region', type=str, default=CONFIG['region'])
    parser.add_argument('--bucket', type=str, default=CONFIG['s3_bucket'])
    parser.add_argument('--role', type=str, default=CONFIG['role_name'])

    # Local options
    parser.add_argument('--data-path', type=str, default=None)

    args = parser.parse_args()

    # Show costs
    if args.costs:
        print_cost_table()
        return

    # Check SageMaker SDK
    if not HAS_SAGEMAKER:
        print("Please install SageMaker SDK: pip install sagemaker boto3")
        sys.exit(1)

    # Create AWS session
    session = boto3.Session(region_name=args.region)
    bucket = get_or_create_bucket(session, args.bucket)

    # Get SageMaker role ARN
    iam = session.client('iam')
    try:
        role_response = iam.get_role(RoleName=args.role)
        role_arn = role_response['Role']['Arn']
    except Exception as e:
        print(f"Could not find IAM role '{args.role}'. Please create it or specify --role")
        print(f"Error: {e}")
        sys.exit(1)

    # Execute action
    if args.setup:
        print("SageMaker setup:")
        print(f"  Region: {args.region}")
        print(f"  Bucket: {bucket}")
        print(f"  Role: {role_arn}")
        print("\nSetup complete! Next steps:")
        print("1. Upload data: python scripts/sagemaker_launcher.py --upload-data")
        print("2. Launch: python scripts/sagemaker_launcher.py --launch")

    elif args.upload_data:
        data_path = args.data_path or Path(__file__).parent.parent / 'data' / 'cmapss'
        upload_data_to_s3(session, bucket, data_path, CONFIG['s3_prefix'])

    elif args.launch:
        # Get wandb API key from argument or environment
        wandb_key = args.wandb_api_key or os.environ.get('WANDB_API_KEY')

        launch_training_job(
            session, bucket, CONFIG['s3_prefix'], role_arn,
            instance_type=args.instance_type,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            spot_instances=args.spot,
            wandb_api_key=wandb_key,
            wandb_project=args.wandb_project,
        )

    elif args.status:
        check_job_status(session)

    elif args.download:
        download_model(session, bucket, args.download, 'outputs/sagemaker_model')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
