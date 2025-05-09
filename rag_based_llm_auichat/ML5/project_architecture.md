# AUIChat Project Architecture

## System Overview

AUIChat is a Retrieval-Augmented Generation (RAG) based application that uses small language models to answer questions about Al Akhawayn University in Ifrane, Morocco. The application integrates document processing, vector embeddings, and language model inference in a modular architecture.

## Component Architecture

```
AUIChat
│
├── Data Processing Layer
│   ├── Document Loader (SimpleDirectoryReader)
│   ├── Text Chunking (SentenceSplitter)
│   └── Text Cleaning (regex-based)
│
├── Vector Storage Layer
│   ├── Qdrant Vector Database
│   │   └── AUIChatVectoreCol-384 Collection
│   └── BGE Embedding Model
│       └── BAAI/bge-small-en-v1.5
│
├── Inference Layer
│   ├── Query Engine
│   │   ├── Vector Similarity Search
│   │   └── Context-Based Response Generation
│   └── Language Models
│       ├── SmolLM-360M-Instruct (primary)
│       └── Mistral-7B-Instruct-v0.3 (alternative)
│
├── API Layer
│   ├── Flask RESTful API
│   │   └── /predict Endpoint
│   └── CORS Support
│
├── User Interface Layer
│   ├── React Frontend
│   └── Responsive UI Components
│
├── Deployment Layer
│   ├── Local Deployment (Seldon Core)
│   │   └── Kubernetes Infrastructure
│   └── Cloud Deployment
│       ├── Google Cloud Run (Backend)
│       └── Firebase Hosting (Frontend)
│
└── CI/CD Pipeline
    ├── GitHub Actions
    ├── Docker Containerization
    └── Automated Testing
```

## Data Flow Architecture

```
[Raw Documents] → [Document Processing] → [Text Chunks] → [Vector Embeddings] → [Qdrant Vector DB]
                                                                                      ↓
[User Query] → [API Endpoint] → [Query Embedding] → [Vector Similarity Search] → [Relevant Chunks]
                                                                                      ↓
                                                                           [Context Augmentation]
                                                                                      ↓
                                             [Response] ← [LLM Inference] ← [Context + Query]
```

## Service Architecture

```
Client Side              |  Cloud Services                     |  Backend Services
--------------------------|-------------------------------------|------------------------
React UI                 |  Firebase Hosting                   |  Flask API Server
(HTML/CSS/JavaScript)    |  Google Cloud Run                   |  SmolLM-360M Model
                         |  Qdrant Cloud                       |  RAG Query Engine
                         |  Google Container Registry          |  BGE Embedding Model
```

## Pipeline Architecture

```
ZenML Pipeline
│
├── Local Pipeline (Seldon/Kubernetes)
│   ├── Data Preparation Phase
│   │   ├── Document Preprocessing
│   │   ├── Data Validation
│   │   └── Vector Index Creation
│   │
│   ├── Model Training Phase
│   │   ├── Model Loading
│   │   ├── Model Saving (MLflow)
│   │   └── MLflow Dashboard Launch
│   │
│   └── Deployment Phase
│       └── Seldon Core Deployment
│
└── Cloud Pipeline (Cloud Run/Firebase)
    ├── Data Preparation Phase
    │   ├── Document Preprocessing
    │   ├── Data Validation
    │   └── Vector Index Creation
    │
    ├── Backend Deployment Phase
    │   ├── RAG App Deployment (Cloud Run)
    │   └── Endpoint Testing
    │
    └── Frontend Deployment Phase
        └── UI Deployment (Firebase)
```

## Code Architecture

```
rag_based_llm_auichat/
│
├── src/ - Core application code
│   ├── main.py - Main entry point, pipeline definitions
│   │
│   ├── data/ - Data processing components
│   │   ├── Data_preprocessing.py - Document loading and chunking
│   │   └── index_storage.py - Qdrant vector index management
│   │
│   ├── engines/ - Core RAG engine
│   │   ├── query_engine.py - Query processing and response generation
│   │   └── local_models/ - Model wrappers for LLMs
│   │
│   ├── workflows/ - Pipeline components
│   │   ├── config/ - Configuration settings
│   │   ├── custom_cloud_run_deployment.py - Cloud Run deployment
│   │   ├── model_saving.py - MLflow model management
│   │   ├── model_training.py - Model training helpers
│   │   ├── evaluation.py - Response evaluation
│   │   └── ui_build.py - UI build process
│   │
│   └── UI/ - Frontend components
│       └── auichat/ - React application
│
├── deployment_scripts/ - Deployment automation
│   ├── deploy_rag_backend_cloudrun.sh - Cloud Run backend deployment
│   └── deploy_ui_cloudrun_firebase.sh - Firebase UI deployment
│
└── .github/workflows/ - CI/CD configuration
    └── auichat-cicd.yml - GitHub Actions workflow
```

## Database Architecture

```
Qdrant Collection: AUIChatVectoreCol-384
│
├── Vector Dimension: 384 (BGE model)
├── Vector Count: ~2218 vectors
│
└── Vector Record Structure:
    ├── id: Unique identifier
    ├── vector: 384-dimension embedding vector
    └── payload:
        ├── text: Document chunk text
        ├── metadata:
        │   ├── file_name: Source document name
        │   └── chunk_id: Position in original document
        └── _node_content: Additional LlamaIndex metadata
```

## API Architecture

```
RESTful API Endpoints:
│
├── POST /predict
│   ├── Request:
│   │   └── {"query": "What are the admission requirements for freshmen?"}
│   │
│   └── Response:
│       └── {"response": "The admission requirements include...", "metadata": {...}}
│
└── GET /health
    └── Response:
        └── {"status": "ok", "version": "1.0"}
```

## Deployment Architecture

```
Development Environment
│
├── Local Development
│   └── ZenML Local Pipeline
│
├── Testing Environment
│   └── GitHub Actions Testing
│
└── Production Environment
    ├── Cloud Run Instance (Backend)
    │   ├── Container: gcr.io/PROJECT_ID/auichat-rag:latest
    │   └── Resources: 2GB RAM, 1 vCPU
    │
    └── Firebase Hosting (Frontend)
        └── URL: https://auichat-988ef.firebaseapp.com/
```

## Security Architecture

```
Authentication & Authorization:
│
├── User Layer: No authentication (public access)
├── Service Layer:
│   ├── Cloud Run: unauthenticated access allowed
│   └── Qdrant Cloud: API key-based authentication
│
└── Resource Access:
    ├── GCP Service Account
    └── Role-based access control
```

## Monitoring Architecture

```
Observability Components:
│
├── Application Logging
│   └── Python logging module
│
├── Performance Metrics
│   ├── MLflow experiment tracking
│   └── Google Cloud Monitoring
│
└── Health Checks
    └── API health endpoint
```

This architecture document provides a comprehensive view of the AUIChat system design, highlighting the modular components, data flows, and deployment patterns. Use this with a text-to-architecture tool to generate visual representations of the system architecture.