# RAG Evaluation with Vertex AI

This directory contains tools for evaluating the AUIChat RAG system using Google Cloud Vertex AI.

## Overview

This implementation provides a centralized approach to evaluate the RAG system using Google Cloud's Vertex AI. The evaluation:

1. Registers your existing Cloud Run RAG service as a model in Vertex AI Model Registry
2. Evaluates key RAG metrics (Context Precision, Context Recall, Faithfulness, Answer Relevancy)
3. Stores evaluation results in Cloud Storage
4. Can be run on-demand or scheduled automatically

## Files

- `vertex_ai_evaluation.py` - Core module with evaluation functions
- `run_vertex_evaluation.py` - Standalone script for running evaluations
- `setup_vertex_evaluation.sh` - Script to set up automated evaluation with Cloud Functions & Scheduler
- `vertex_ai_requirements.txt` - Python dependencies for Vertex AI integration

## Integration with ZenML

A step has been added to the cloud deployment pipeline in `src/workflows/vertex_evaluation_step.py` that automatically runs evaluation after deployment.

## Usage

### Option 1: Run On-Demand Evaluation

```bash
# Basic usage with default endpoint
python run_vertex_evaluation.py

# Specify custom endpoint and project
python run_vertex_evaluation.py --endpoint https://your-endpoint.run.app/predict --project your-project-id

# Save results to a file
python run_vertex_evaluation.py --output results.json

# Complete evaluation with both standard metrics and adversarial testing
python vertex_ai_evaluation_complete.py --endpoint https://your-endpoint.run.app/predict --project your-project-id --output complete_evaluation.json
```

### Option 2: Setup Automated Evaluation

```bash
# Set up Cloud Function and Scheduler
./setup_vertex_evaluation.sh
```

This will:
1. Create a Cloud Function that runs the evaluation
2. Set up a Cloud Scheduler job to run daily evaluations
3. Create a service account with necessary permissions
4. Create a Cloud Storage bucket for results if it doesn't exist

### Option 3: Use the ZenML Pipeline

The evaluation is integrated into the cloud deployment pipeline and will run automatically after model deployment.

## Metrics Explanation

1. **Context Precision**: Measures how relevant the retrieved contexts are to the question
2. **Context Recall**: Assesses how well the retrieved contexts cover the information in the reference answer
3. **Faithfulness**: Evaluates whether the generated answer is factually consistent with the retrieved contexts
4. **Answer Relevancy**: Measures how well the answer addresses the original question

## Viewing Results

Results are stored in Cloud Storage bucket: `gs://auichat-rag-metrics/`

You can view the latest results with:
```bash
gsutil cat gs://auichat-rag-metrics/latest_vertex_ai_evaluation.json
```

## Current and Future Extensions

This implementation includes:

1. **Adversarial Testing**: Evaluation of model robustness against inappropriate requests, prompt injections, and edge cases

And can be further extended to support:

2. **A/B Testing**: Compare two RAG model versions side by side
3. **Bias Detection**: Test for potential biases in responses
4. **Advanced Robustness Testing**: Expand test suite with more categories of adversarial inputs
