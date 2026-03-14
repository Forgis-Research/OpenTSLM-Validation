# OpenTSLM Validation Guide

Reproduce the OpenTSLM paper results using pretrained models on SageMaker.

## Paper Metrics to Reproduce

| Task | Metric | Paper Result | Test Samples |
|------|--------|--------------|--------------|
| TSQA | Accuracy | 97.50% | 4,800 |
| HAR-CoT | F1 | 65.44% | 8,222 |
| Sleep-CoT | F1 | 69.88% | 930 |
| ECG-QA-CoT | F1 | 40.25% | 41,093 |

## Prerequisites

### 1. HuggingFace Authentication (Required for Llama/Gemma)

```bash
# Request Llama access: https://huggingface.co/meta-llama/Llama-3.2-1B
# Create token: https://huggingface.co/settings/tokens

# Login
huggingface-cli login
# Or set environment variable
export HF_TOKEN="your_token_here"
```

### 2. AWS Setup for SageMaker

```bash
# Install AWS CLI and configure
pip install awscli boto3 sagemaker
aws configure

# Verify
aws sts get-caller-identity
```

### 3. Create SageMaker Role (if not exists)

1. Go to IAM Console → Roles → Create Role
2. Select "SageMaker" as trusted entity
3. Attach policy: `AmazonSageMakerFullAccess`
4. Name it: `SageMakerExecutionRole`

## Usage

### Option A: Quick Local Test (No SageMaker)

```bash
cd OpenTSLM

# Install OpenTSLM locally
pip install -e original_opentslm

# Quick test (5 samples)
python scripts/validate_opentslm.py --task tsqa --quick --max-samples 5

# Full local validation (if you have GPU)
python scripts/validate_opentslm.py --all
```

### Option B: SageMaker Validation (Recommended)

```bash
# Check setup
python scripts/sagemaker_validation.py --setup

# View cost estimates
python scripts/sagemaker_validation.py --costs

# Launch quick validation (~$0.50, 30 min)
python scripts/sagemaker_validation.py --launch --quick

# Launch full validation (~$2-3, 2-3 hours)
python scripts/sagemaker_validation.py --launch

# Check status
python scripts/sagemaker_validation.py --status

# Download results
python scripts/sagemaker_validation.py --download <job-name>
```

### Option C: Spot Instances (60-70% cheaper)

```bash
# Use spot for full validation (~$1)
python scripts/sagemaker_validation.py --launch --spot
```

## Cost Estimates

| Mode | Instance | Time | Cost |
|------|----------|------|------|
| Quick (100 samples/task) | ml.g4dn.xlarge | ~30 min | ~$0.50 |
| Full (~470K samples) | ml.g4dn.xlarge | ~3 hours | ~$2.50 |
| Full with Spot | ml.g4dn.xlarge | ~3 hours | ~$0.80 |

## Available Models (51 checkpoints)

```
# Model naming: OpenTSLM/{llm}-{task}-{type}
# Examples:
OpenTSLM/llama-3.2-1b-tsqa-sp
OpenTSLM/llama-3.2-1b-har-flamingo
OpenTSLM/gemma-3-270m-sleep-sp
OpenTSLM/llama-3.2-3b-ecg-flamingo
```

## Validation Output

Results are saved to `outputs/validation/`:

```json
{
  "task": "tsqa",
  "metrics": {"accuracy": 97.2},
  "paper_value": 97.5,
  "difference": -0.3,
  "status": "PASS"
}
```

## Troubleshooting

### "Model not found" Error
- Check HuggingFace authentication: `huggingface-cli whoami`
- Request Llama access if needed

### "Out of memory" Error
- Use SoftPrompt (sp) instead of Flamingo
- Reduce batch size
- Use larger instance (ml.p3.2xlarge)

### Slow Inference
- Use GPU instance (ml.g4dn.xlarge minimum)
- Avoid CPU inference for full validation

## Next Steps After Validation

1. **If results match paper (±5%)**: Proceed to industrial extension
2. **If results differ**: Check model versions, authentication
3. **Document discrepancies**: Note any differences for paper

## Files

- `scripts/validate_opentslm.py` - Main validation script
- `scripts/sagemaker_validation.py` - SageMaker launcher
- `scripts/sagemaker_entry.py` - SageMaker entry point
- `original_opentslm/` - Cloned OpenTSLM repository
