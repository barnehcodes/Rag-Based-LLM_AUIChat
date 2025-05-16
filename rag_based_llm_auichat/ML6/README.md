# AUI Chat RAG - Enhanced Deployment

This directory contains tools for deploying and testing the enhanced version of the AUIChat RAG system. This implementation focuses on a standalone deployment with all advanced RAG features.

## Overview

The enhanced RAG system includes the following improvements:

1. **Hybrid Retrieval**: Combines vector-based and BM25 keyword-based retrieval for better results
2. **Query Reformulation**: Improves query understanding with T5-based reformulation
3. **Advanced Re-ranking**: Uses cross-encoder models to improve search result ranking
4. **Multi-hop RAG**: Performs follow-up queries for complex questions
5. **Improved Prompting**: Better instructions for the LLM
6. **Generation Parameters**: Lower temperature (0.3) for improved factuality

## Files

- `deploy_version_b.py` - Script to deploy Version B to Google Cloud Run
- `test_version_b_locally.py` - Script to test Version B without deploying

## Deployment Instructions

### Option 1: Deploy Version B to Cloud Run

```bash
# Deploy Version B to Cloud Run
python deploy_version_b.py --preprocessed-nodes /home/barneh/Rag-Based-LLM_AUIChat/preprocessed_nodes.pkl

# Specify a custom service name
python deploy_version_b.py --service-name auichat-rag-custom-name --preprocessed-nodes /path/to/preprocessed_nodes.pkl

# Specify a different project and region
python deploy_version_b.py --project-id your-project-id --region us-west1 --preprocessed-nodes /path/to/preprocessed_nodes.pkl
```

### Option 2: Test Version B Locally

```bash
# Test the Version B enhancements locally before deployment
python test_version_b_locally.py
```

## Calling the API

Once deployed, you can call the API using curl:

```bash
curl -X POST https://auichat-rag-version-b-[HASH].a.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the admission requirements for AUI?"}'
```

## Implementation Notes

### Local Docker Testing

Before deploying to Cloud Run, you can test the deployment process locally:

```bash
# 1. Create a local testing directory
mkdir -p /tmp/rag_deployment_test

# 2. Run the deployment script with --local-only flag
python deploy_version_b.py --preprocessed-nodes /path/to/preprocessed_nodes.pkl --local-only

# 3. Build and run the container locally
cd /tmp/rag_deployment_test
docker build -t rag-version-b-test .
docker run -p 8080:8080 -e PREPROCESSED_NODES=/app/preprocessed_nodes.pkl rag-version-b-test

# 4. Test the locally running container
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"query": "What is AUI?"}'
```

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
