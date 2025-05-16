# A/B Testing for AUIChat RAG System

This module provides tools for comparing two RAG system implementations using the same metrics and test data.

## Overview

The A/B testing system allows you to:

1. Compare a current production RAG endpoint (System A) with a modified version (System B)
2. Evaluate both systems using the same test questions and evaluation metrics
3. Generate detailed reports showing which system performs better
4. Store results in Google Cloud Storage for tracking over time

## Files

- `ab_testing.py` - Core module with comparison logic and metrics calculation
- `run_ab_testing.py` - Standalone script for on-demand A/B testing
- `run_ab_testing_pipeline.py` - ZenML pipeline for A/B testing
- `ab_testing_step.py` - ZenML step for integration with deployment pipelines

## Integration with ZenML

A/B testing is integrated into the cloud deployment pipeline in `src/workflows/ab_testing_step.py` and can automatically compare new deployments to a baseline.

## Usage

### Option 1: Run On-Demand A/B Testing

```bash
# Basic usage comparing two endpoints
python ML6/run_ab_testing.py \
  --endpoint-a https://current-endpoint.run.app/predict \
  --endpoint-b https://modified-endpoint.run.app/predict

# Save results to a file
python ML6/run_ab_testing.py \
  --endpoint-a https://current-endpoint.run.app/predict \
  --endpoint-b https://modified-endpoint.run.app/predict \
  --output ab_results.json

# Don't store results in GCS
python ML6/run_ab_testing.py \
  --endpoint-a https://current-endpoint.run.app/predict \
  --endpoint-b https://modified-endpoint.run.app/predict \
  --no-gcs
```

### Option 2: Run Standalone A/B Testing Pipeline

```bash
# Run as ZenML pipeline
python ML6/run_ab_testing_pipeline.py \
  --current-endpoint https://current-endpoint.run.app \
  --modified-endpoint https://modified-endpoint.run.app
```

### Option 3: Automatic A/B Testing in Deployment Pipeline

The deployment pipeline in `src/main.py` will automatically run A/B testing if a baseline endpoint is specified:

```bash
# Set baseline endpoint before running pipeline
export AUICHAT_BASELINE_ENDPOINT="https://baseline-endpoint.run.app"

# Run cloud deployment pipeline
python src/main.py cloud
```

## Metrics Explanation

A/B testing evaluates both systems using these metrics:

1. **Context Precision**: Measures how relevant the retrieved contexts are to the question
2. **Context Recall**: Assesses how well the retrieved contexts cover the information in the reference answer
3. **Faithfulness**: Evaluates whether the generated answer is factually consistent with the retrieved contexts
4. **Answer Relevancy**: Measures how well the answer addresses the original question
5. **Overall Score**: Weighted combination of the above metrics

## Viewing Results

Results are stored in Google Cloud Storage bucket: `gs://auichat-rag-metrics/`

You can view the latest results with:
```bash
gsutil cat gs://auichat-rag-metrics/latest_ab_testing.json
```

## Example Implementation of System B (Improved RAG)

An improved RAG system might include:

1. Hybrid retrieval combining vector and keyword search
2. Query reformulation to improve retrieval accuracy
3. Multi-hop RAG architecture for complex questions
4. Advanced re-ranking of retrieved contexts 
5. Improved prompt engineering
6. Fine-tuned generation parameters

See `run_ab_testing.py` for more detailed examples and instructions.
