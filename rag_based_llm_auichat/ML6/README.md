# AUI Chat RAG - Evaluation Framework

This directory contains evaluation tools for assessing the performance, robustness, and safety of the AUIChat RAG system. The evaluation framework provides comprehensive metrics and tests to ensure the deployed system meets quality standards before and after deployment.

## Overview

The evaluation framework includes:

1. **Vertex AI Integration**: Tools for registering and evaluating the RAG system using Google Cloud's Vertex AI
2. **Customized Metrics**: RAG-specific metrics for faithfulness, context precision/recall, and answer relevancy
3. **Robustness Testing**: Adversarial query testing to evaluate system safety
4. **Evaluation Pipeline**: Integration with deployment workflows to automate evaluation
5. **Enhanced Deployment**: Tools for deploying improved RAG versions with advanced features

## Files

- `vertex_ai_evaluation.py` - Main evaluation logic and metrics implementation
- `run_vertex_evaluation.py` - Command-line tool to run evaluations
- `deploy_version_b.py` - Script to deploy enhanced RAG version to Cloud Run
- `test_version_b_locally.py` - Script to test enhanced RAG without deploying
- `vertex_ai_requirements.txt` - Dependencies for evaluation

## Evaluation Metrics

The framework implements several specialized metrics for RAG system evaluation:

### Context Precision (`calculate_context_precision()`)

Measures how relevant the retrieved contexts are to the original question.

```python
def calculate_context_precision(question: str, contexts: List[Dict]) -> float:
    """
    Calculation method:
    1. Extract all texts from context objects
    2. Generate question keywords 
    3. Count contexts that contain at least one question keyword
    4. Calculate: relevant_contexts / total_contexts
    """
```

### Context Recall (`calculate_context_recall()`)

Measures how well the retrieved contexts cover information in the reference answer.

```python
def calculate_context_recall(reference: str, contexts: List[Dict]) -> float:
    """
    Calculation method:
    1. Extract all texts from context objects
    2. Generate reference answer keywords
    3. Count reference words that appear in at least one context
    4. Calculate: covered_words / total_reference_words
    """
```

### Faithfulness (`calculate_faithfulness()`)

Quantifies how much of the response is supported by the retrieved contexts.

```python
def calculate_faithfulness(response: str, contexts: List[Dict]) -> float:
    """
    Calculation method:
    1. Extract all texts from context objects
    2. Create a set of words from all contexts
    3. Count response words that appear in the context word set
    4. Calculate: supported_words / total_response_words
    
    A higher score indicates the response stays closer to information
    in the retrieved contexts, reducing hallucination.
    """
```

### Answer Relevancy (`calculate_answer_relevancy()`)

Measures how relevant the response is to the original question.

```python
def calculate_answer_relevancy(question: str, response: str) -> float:
    """
    Calculation method:
    1. Generate sets of question and response words
    2. Calculate overlap between the two sets
    3. Calculate: overlap_words / total_question_words
    """
```

## Robustness Testing

The framework includes comprehensive adversarial testing to evaluate how well the system handles inappropriate or problematic queries:

```python
def evaluate_robustness(endpoint_url: str) -> Dict[str, Any]:
    """
    Tests the system against adversarial inputs like:
    - Requests for unauthorized system access
    - Personal/private information requests
    - Prompt injection attempts
    - Requests to generate harmful content
    
    For each test, the system checks if the response appropriately
    refuses to answer based on keyword detection.
    
    The robustness score is calculated as:
    correct_responses / total_adversarial_tests
    """
```

## Notebook: `eval_try.ipynb`

Located in the `notebooks` directory, this notebook implements a comprehensive evaluation framework that goes beyond traditional accuracy metrics, focusing on three critical dimensions:

### 1. Bias Auditing
The notebook implements rigorous bias testing to identify and measure potential biases across demographic groups:
- **Demographic Analysis**: Tests system performance across different nationalities, genders, and age groups
- **Disparity Metrics**: Quantifies performance gaps between reference groups and others
- **Intersectional Analysis**: Examines how combinations of attributes (e.g., nationality and gender) affect performance
- **Visualization**: Creates custom disparity plots and heatmaps to highlight potential biases

### 2. Robustness Testing
Implements a systematic approach to evaluate how the RAG system handles challenging inputs:
- **Perturbation Testing**: Generates variations of queries with typos, word order changes, and synonym replacements
- **Adversarial Queries**: Tests with deliberately confusing or contradictory questions
- **Edge Case Handling**: Evaluates performance with unusual inputs (very long/short queries, special characters)
- **Performance Degradation Analysis**: Measures how different query modifications impact relevance scores

Sample perturbations generated include:
```python
# Original: "What are the housing options for students at AUI?"
# Typo: "What are the housinf options for students at AUI?"
# Word order: "What are options the housing for students at AUI?"
# Synonym: "What are the accommodation options for students at AUI?"
# Case: "what are the housing options for students at aui?"
```

### 3. Model Explainability
Implements techniques to interpret and explain how the RAG system makes decisions:
- **Retrieval Analysis**: Visualizes how different contexts contribute to the answer
- **Feature Importance**: Uses LIME to explain which words/features influence query processing
- **Attribution Scoring**: Measures how the final answer draws information from different contexts
- **Transparency Visualization**: Creates charts showing context relevance and answer overlap

### Implementation Details
The notebook uses a modular approach with specialized functions for different evaluation aspects:
- `generate_perturbations()`: Creates linguistic variations of baseline queries
- `generate_adversarial_queries()`: Creates challenging test cases to stress-test the system
- `analyze_robustness_results()`: Calculates performance metrics across different test types
- `process_rag_response_for_explanation()`: Extracts relevant data for explainability analysis
- `attribution_analysis()`: Examines how generated answers relate to source contexts

The framework provides a comprehensive evaluation dashboard that summarizes performance across all three dimensions, helping identify specific areas for improvement in the RAG system.

## Usage

### Running Vertex AI Evaluation

```bash
# Basic evaluation with default settings
python run_vertex_evaluation.py

# Specify custom endpoint and project
python run_vertex_evaluation.py --endpoint https://your-endpoint.run.app/predict --project your-project-id

# Save results to a file
python run_vertex_evaluation.py --output results.json

# Complete evaluation with both standard metrics and adversarial testing
python vertex_ai_evaluation_complete.py --endpoint https://your-endpoint.run.app/predict --project your-project-id --output complete_evaluation.json
```

### Enhanced RAG Deployment

#### Option 1: Deploy Version B to Cloud Run

```bash
# Deploy Version B to Cloud Run
python deploy_version_b.py --preprocessed-nodes /home/barneh/Rag-Based-LLM_AUIChat/preprocessed_nodes.pkl

# Specify a custom service name
python deploy_version_b.py --service-name auichat-rag-custom-name --preprocessed-nodes /path/to/preprocessed_nodes.pkl

# Specify a different project and region
python deploy_version_b.py --project-id your-project-id --region us-west1 --preprocessed-nodes /path/to/preprocessed_nodes.pkl
```

#### Option 2: Test Version B Locally

```bash
# Test the Version B enhancements locally before deployment
python test_version_b_locally.py
```

### Setting Up Automated Evaluation

The evaluation can be set up to run automatically:

```bash
# Set up Cloud Function and Scheduler
./setup_vertex_evaluation.sh
```

This will:
1. Create a Cloud Function that runs the evaluation
2. Set up a Cloud Scheduler job to run daily evaluations
3. Create a service account with necessary permissions
4. Create a Cloud Storage bucket for results if it doesn't exist

#### ZenML Pipeline Integration

The evaluation is integrated into the cloud deployment pipeline and will run automatically after model deployment.

## Implementation Notes

The evaluation framework follows these best practices:

1. **Test Data Separation**: Uses distinct test sets for different evaluation aspects
2. **Comprehensive Metrics**: Goes beyond accuracy to measure RAG-specific qualities
3. **Safety Checks**: Includes adversarial testing to ensure system safety
4. **Result Storage**: Saves evaluation results to Cloud Storage for tracking
5. **Integration**: Works with both development and production environments

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

## Contributing

When extending the evaluation framework:

1. Add new test cases to the appropriate test data sets
2. Implement new metrics in the `vertex_ai_evaluation.py` file
3. Update the notebook for comprehensive evaluations
4. Document the metrics calculation methodology

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
