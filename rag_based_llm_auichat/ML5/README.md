# AUIChat Application: ML5 Documentation

This document provides comprehensive documentation for the AUIChat application development, integration, and deployment processes, with a focus on the two pipeline options (local and cloud-based).

## Table of Contents

1. [Application Overview](#application-overview)
2. [Application Development](#application-development)
   - [Serving Modes Implementation](#serving-modes-implementation)
   - [Model Service Development](#model-service-development)
   - [Front-end Client Development](#front-end-client-development)
3. [Integration and Deployment](#integration-and-deployment)
   - [Packaging and Containerization](#packaging-and-containerization)
   - [CI/CD Pipeline Integration](#cicd-pipeline-integration)
   - [Hosting Options](#hosting-options)
4. [Model Serving](#model-serving)
   - [Model Serving Runtime](#model-serving-runtime)
5. [Deployment Pipelines](#deployment-pipelines)
   - [Local Deployment Pipeline](#local-deployment-pipeline)
   - [Cloud Deployment Pipeline](#cloud-deployment-pipeline)
6. [Testing the Application](#testing-the-application)
7. [File Placement Guide](#file-placement-guide)
8. [Command Reference](#command-reference)

## Application Overview

AUIChat is a Retrieval-Augmented Generation (RAG) based application that uses small language models (SmolLM) to answer questions about Al Akhawayn University in Ifrane, Morocco. The application leverages Qdrant vector database for efficient similarity search and implements MLflow for experiment tracking and model management.

Live demo: [https://auichat-988ef.firebaseapp.com/](https://auichat-988ef.firebaseapp.com/)
## Application Architecture 
[arch](./arch.png)
## Application Development

### Serving Modes Implementation

AUIChat implements three serving modes to address different use cases:

1. **On-demand to human (Interactive)**:
   - Implemented through a React-based web UI that allows users to submit queries and receive responses in real-time.
   - File location: `/rag_based_llm_auichat/src/UI/auichat/`

2. **On-demand to machine (API)**:
   - RESTful API endpoint that accepts JSON requests and returns structured responses.
   - File location: `/improved_rag_app_qdrant.py`
   - API Endpoint format:
     ```
     POST /predict
     {
       "query": "What are the admission requirements for freshmen?"
     }
     ```

3. **Batch processing**:
   - ZenML pipeline steps that can process multiple queries in batch mode.
   - File location: `/rag_based_llm_auichat/src/engines/query_engine.py`
   - Used primarily for testing and evaluation purposes.

### Model Service Development

The model service architecture consists of several components:

1. **Data Preprocessing**:
   - Document loading, cleaning, and chunking
   - File location: `/rag_based_llm_auichat/src/data/Data_preprocessing.py`
   
2. **Vector Storage**:
   - Integration with Qdrant vector database for efficient similarity search
   - Configuration: `/rag_based_llm_auichat/src/workflows/config/config.py`
   
3. **Query Engine**:
   - Retrieval component that finds relevant document chunks
   - LLM component that generates responses based on retrieved context
   - File location: `/rag_based_llm_auichat/src/engines/query_engine.py`

4. **Model Serving**:
   - Local Seldon Core deployment for Kubernetes environments
   - Cloud Run deployment for cloud environments
   - Files: 
     - `/rag_based_llm_auichat/src/workflows/model_saving.py`
     - `/rag_based_llm_auichat/src/workflows/custom_cloud_run_deployment.py`

### Front-end Client Development

The front-end client is built using React and Vite, providing a responsive and intuitive interface for users:

1. **UI Components**:
   - Chat interface with message history
   - Input form for user queries
   - Loading indicators and error handling
   - File location: `/rag_based_llm_auichat/src/UI/auichat/`

2. **API Integration**:
   - Communication with backend API endpoints
   - Handling of asynchronous responses
   - Environment-based configuration for development and production

3. **Responsive Design**:
   - Mobile-friendly layouts
   - Accessibility considerations

## Integration and Deployment

### Packaging and Containerization

Both the backend and frontend components are containerized using Docker:

1. **Backend Container**:
   - Python 3.10 base image
   - Required dependencies installed via pip
   - Gunicorn as WSGI server
   - File location: `/deployment_scripts/deploy_rag_backend_cloudrun.sh`

2. **Frontend Container**:
   - Nginx base image for serving static files
   - Build artifacts from React application
   - CORS configuration for API communication
   - File location: `/deployment_scripts/deploy_ui_cloudrun_firebase.sh`

### CI/CD Pipeline Integration

The application is integrated with GitHub Actions for continuous integration and deployment:

1. **Workflow Configuration**:
   - Automated testing, building, and deployment
   - Support for multiple environments (dev, staging, production)
   - File location: `/.github/workflows/auichat-cicd.yml`

2. **Pipeline Stages**:
   - Lint and test: Runs unit tests and linting
   - Build: Creates Docker images for backend and frontend
   - Deploy: Deploys to either local Kubernetes or GCP Cloud Run
   - Notify: Sends notifications of deployment status

3. **Trigger Events**:
   - Push to main branch: Full pipeline execution
   - Pull request to main: Tests only
   - Manual trigger: User-selected deployment target

### Hosting Options

The application supports two hosting scenarios:

1. **Local Kubernetes Deployment**:
   - Using Seldon Core for model serving
   - Local storage for vector database
   - MLflow for experiment tracking and model registry

2. **Cloud Deployment**:
   - Google Cloud Run for backend services
   - Firebase Hosting for frontend UI
   - Qdrant Cloud for vector database
   - GCP Artifact Registry for container images

## Model Serving

### Model Serving Runtime

The application uses multiple serving runtimes depending on the deployment scenario:

1. **Seldon Core (Local)**:
   - Kubernetes-based model serving platform
   - MLflow model format support
   - Horizontal scaling capabilities
   - File: `/rag_based_llm_auichat/src/main.py` (LOCAL_AUICHAT_DEPLOYMENT_PIPELINE)

2. **Cloud Run (Cloud)**:
   - Serverless container runtime on GCP
   - Auto-scaling based on demand
   - Authentication and security features
   - File: `/rag_based_llm_auichat/src/workflows/custom_cloud_run_deployment.py`

3. **MLflow Tracking**:
   - Experiment tracking for model development
   - Model registry for versioning
   - Deployment metadata storage
   - UI for visualization and comparison
   - File: `/rag_based_llm_auichat/src/workflows/mlflow_utils.py`

## Deployment Pipelines

### Local Deployment Pipeline

[local_pipeline](./pipline_local.png)

The local deployment pipeline uses Seldon Core and Kubernetes for deployment:

1. **Pipeline Structure** (defined in `src/main.py`):
   - **BIG STEP 1: DATA_ACQUISITION_VALIDATION_AND_PREPARATION**
     - Preprocess data (load, clean, chunk documents)
     - Validate processed data
     - Create and store index in Qdrant
     - Validate Qdrant storage

   - **BIG STEP 2: MODEL_TRAINING_AND_EVALUATION**
     - Placeholder model training
     - Save model using MLflow
     - Launch MLflow dashboard

   - **BIG STEP 3: ML_PRODUCTIONIZATION**
     - Deploy to Seldon Core
     - Launch UI components

2. **Commands to Execute**:
   ```bash
   # Run the local deployment pipeline
   python src/main.py local

   # Or with environment variable
   AUICHAT_PIPELINE_CHOICE=local python src/main.py
   ```

3. **Requirements**:
   - Kubernetes cluster with Seldon Core installed
   - ZenML with Seldon integration configured
   - MLflow tracking server
   - Local UI server

### Cloud Deployment Pipeline

The cloud deployment pipeline uses Google Cloud Run and Firebase:

1. **Pipeline Structure** (defined in `src/main.py`):
   - **BIG STEP: DATA_PREPARATION_FOR_CLOUD_ENVIRONMENT**
     - Preprocess data (load, clean, chunk documents)
     - Validate processed data
     - Create and store index in Qdrant
     - Validate Qdrant storage

   - **BIG STEP: DEPLOY_RAG_BACKEND_TO_CLOUD_RUN**
     - Deploy the improved_rag_app to Cloud Run

   - **BIG STEP: TEST_CLOUD_RUN_ENDPOINT**
     - Test the deployed endpoint via POST request

   - **BIG STEP: BUILD_AND_DEPLOY_UI_TO_FIREBASE**
     - Build UI if needed
     - Deploy to Firebase hosting

2. **Commands to Execute**:
   ```bash
   # Run the cloud deployment pipeline
   python src/main.py cloud

   # Or with environment variable
   AUICHAT_PIPELINE_CHOICE=cloud python src/main.py
   ```

3. **Requirements**:
   - GCP project with Cloud Run and Firebase enabled
   - Appropriate permissions (Service Account)
   - Qdrant Cloud instance configured
   - Environment variables for authentication

## Testing the Application

### Local Testing

```bash
# Test the local deployment
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"query": "What are the admission requirements for freshmen?"}'

# Access the MLflow UI
http://localhost:5001
```

### Cloud Testing

```bash
# Test the Cloud Run deployment
curl -X POST https://auichat-rag-prod-[hash].a.run.app/predict -H "Content-Type: application/json" -d '{"query": "What are the admission requirements for freshmen?"}'

# Access the web UI
https://auichat-988ef.firebaseapp.com/
```

## File Placement Guide

### Key Files and Their Locations

```
/rag_based_llm_auichat/
├── src/
│   ├── main.py                   # Main application with both deployment pipelines
│   ├── data/
│   │   ├── Data_preprocessing.py # Data preprocessing step
│   │   └── index_storage.py      # Qdrant index creation and storage
│   ├── engines/
│   │   └── query_engine.py       # RAG query engine implementation
│   ├── workflows/
│   │   ├── config/
│   │   │   └── config.py         # Configuration for Qdrant and embeddings
│   │   ├── cloud_testing.py      # Cloud Run endpoint testing
│   │   ├── custom_cloud_run_deployment.py # Cloud Run deployment
│   │   ├── data_validation.py    # Data validation step
│   │   ├── mlflow_utils.py       # MLflow dashboard launcher
│   │   ├── model_saving.py       # Model saving with MLflow
│   │   ├── model_training.py     # Placeholder model training
│   │   └── ui_build.py           # UI building step
│   └── UI/
│       └── auichat/              # React UI implementation
├── ML5/
│   └── README.md                 # This documentation file
├── notebooks/
│   └── experiments.ipynb         # Experimental notebook with embedding comparisons
├── deployment_scripts/
│   ├── deploy_rag_backend_cloudrun.sh # Backend deployment script
│   └── deploy_ui_cloudrun_firebase.sh # UI deployment script
└── .github/
    └── workflows/
        └── auichat-cicd.yml      # GitHub Actions CI/CD workflow
```

### External Files

```
/home/barneh/Rag-Based-LLM_AUIChat/
├── improved_rag_app_qdrant.py    # Improved RAG application for Cloud Run
├── raw/                          # Raw document files for processing
│   ├── AUI Catalog_2023-2024_New_Version.pdf
│   ├── Counseling Services FAQ Spring 2024.pdf
│   └── ...
├── preprocessed_nodes.pkl        # Preprocessed document nodes
├── cloudrun_deployment_info.json # Cloud Run deployment information
└── cloudrun_qdrant_info.json     # Qdrant configuration information
```

## Command Reference

### Development Commands

```bash
# Install dependencies
pip install -e rag_based_llm_auichat/

# Run data preprocessing separately
python -c "from rag_based_llm_auichat.src.data.Data_preprocessing import preprocess_data; preprocess_data()"

# Build UI manually
cd rag_based_llm_auichat/src/UI/auichat && npm install && npm run build
```

### Deployment Commands

```bash
# Local deployment pipeline
python rag_based_llm_auichat/src/main.py local

# Cloud deployment pipeline
python rag_based_llm_auichat/src/main.py cloud

# Deploy backend manually
bash deployment_scripts/deploy_rag_backend_cloudrun.sh

# Deploy UI manually
bash deployment_scripts/deploy_ui_cloudrun_firebase.sh
```

### CI/CD Commands

```bash
# Trigger GitHub Actions workflow manually
# (Via GitHub UI or API)

# Test a specific pipeline step
python -c "from zenml.integrations.seldon.steps import seldon_model_deployer_step; help(seldon_model_deployer_step)"

# Run MLflow dashboard manually
python -c "from rag_based_llm_auichat.src.workflows.mlflow_utils import launch_mlflow_dashboard_step; launch_mlflow_dashboard_step()"
```

### Additional Resources

- Demo Site: [https://auichat-988ef.firebaseapp.com/](https://auichat-988ef.firebaseapp.com/)
- Documentation for ZenML: [https://docs.zenml.io/](https://docs.zenml.io/)
- Documentation for Seldon Core: [https://docs.seldon.io/](https://docs.seldon.io/)
- Documentation for Cloud Run: [https://cloud.google.com/run/docs](https://cloud.google.com/run/docs)
