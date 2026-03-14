# OpenTSLM Validation & Industrial Extension

Reproducing and extending OpenTSLM (Time-Series Language Models) for industrial applications.

## Goals

1. **Reproduce** original OpenTSLM paper results
2. **Fix weaknesses** identified in the original work
3. **Extend to industrial domain** (FactoryNet, predictive maintenance)
4. **Prove transferability** across machine types

## Paper Reference

- **Original Paper**: [OpenTSLM: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data](https://arxiv.org/abs/2510.02410)
- **Original Repo**: [StanfordBDHG/OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM)

## Paper Metrics to Reproduce

| Task | Metric | Paper Result | Test Samples |
|------|--------|--------------|--------------|
| TSQA | Accuracy | 97.50% | 4,800 |
| HAR-CoT | F1 | 65.44% | 8,222 |
| Sleep-CoT | F1 | 69.88% | 930 |
| ECG-QA-CoT | F1 | 40.25% | 41,093 |

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -e original_opentslm
pip install sagemaker boto3 scikit-learn

# HuggingFace login (for model access)
huggingface-cli login
```

### Run Validation

```bash
# Local quick test (5 samples)
python scripts/validate_opentslm.py --task tsqa --llm gemma-3-270m --max-samples 5

# SageMaker validation (~$0.50, 30 min)
python scripts/sagemaker_validation.py --launch --quick --llm gemma-3-270m

# Full validation (~$2.50, 3 hours)
python scripts/sagemaker_validation.py --launch --llm gemma-3-270m
```

### Check Status & Download Results

```bash
python scripts/sagemaker_validation.py --status
python scripts/sagemaker_validation.py --download <job-name>
```

## Repository Structure

```
.
├── scripts/
│   ├── validate_opentslm.py      # Main validation script
│   ├── sagemaker_validation.py   # SageMaker launcher
│   ├── sagemaker_entry.py        # SageMaker entry point
│   └── VALIDATION_README.md      # Detailed validation guide
├── original_opentslm/            # Original OpenTSLM code (MIT License)
└── README.md
```

## Available Models (51 pretrained checkpoints)

Models available on HuggingFace: `OpenTSLM/{llm}-{task}-{type}`

| LLM | Tasks | Types |
|-----|-------|-------|
| llama-3.2-1b | tsqa, m4, har, sleep, ecg | sp, flamingo |
| llama-3.2-3b | tsqa, m4, har, sleep, ecg | sp, flamingo |
| gemma-3-270m | tsqa, m4, har, sleep, ecg | sp, flamingo |
| gemma-3-1b-pt | tsqa, m4, har, sleep, ecg | sp, flamingo |

## License

- This repository: MIT License
- Original OpenTSLM: MIT License

## Acknowledgments

Based on work by Stanford Biodesign Digital Health Group and ETH Zurich.
