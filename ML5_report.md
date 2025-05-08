# Milestone 5: ML Productionization - AUIChat System Report

## 1. Introduction

This report details the implementation of Milestone 5 (ML Productionization) for the AUIChat project, focusing on the development, packaging, deployment, and serving aspects of the machine learning system. We have created a production-ready RAG-based chatbot system that can be deployed through multiple cloud services and delivery methods.

## 2. ML System Architecture

The AUIChat system follows a modular architecture designed for scalability and maintainability. The architecture is defined in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/m1_architecture/`.

### Key Architecture Components:

- **ZenML Pipelines**: Core orchestration layer that manages all deployment workflows
- **RAG Engine**: Retrieval-augmented generation system combining vector similarity search with LLM inference
- **UI Layer**: React-based frontend for user interaction
- **Deployment Options**: Multiple cloud deployment targets (Seldon Core, Vertex AI, Cloud Run)

## 3. Model Serving Mode

AUIChat implements an on-demand serving model designed specifically for chat interactions:

### Implementation Details:

- **Serving Mode**: Interactive query-response (chatbot) mode
- **Vector Database**: Qdrant for storing and retrieving document embeddings
- **LLM Engine**: SmolLM-360M-Instruct model for generating contextual responses
- **Implementation**: Found in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/engines/query_engine.py`

The system handles chat requests synchronously, with the query engine retrieving relevant context from Qdrant before sending it to the LLM for response generation.

## 4. Model Service Development

The model service components are distributed across multiple files in the codebase:

### Frontend Client:

- **UI Launcher**: `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/m5_frontend_client/ui_launcher.py`
  - Orchestrates starting the Flask API server, CORS proxy, and React frontend
  - Creates a seamless development environment for testing the application
  - Runs as a ZenML step in the deployment pipeline

### Backend Services:

- **Query Engine**: `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/engines/query_engine.py`
  - Implements the RAG logic for retrieving and generating responses
  - Configures the Qdrant vector store and embedding model

- **Local Model Handler**: `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/engines/local_models/model_loader.py`
  - Loads and configures the SmolLM-360M model for local inference
  - Provides a text-generation pipeline for the RAG system

## 5. Model Serving Runtime

The AUIChat system employs a flexible approach to model serving, with options for both local and cloud-based deployment:

### MLflow Serving:

- **Model Saving**: `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/workflows/model_saving.py`
  - Packages the SmolLM-360M-Instruct model using MLflow
  - Handles model artifact creation for deployment targets
  - Creates standardized model versions for tracking

### ZenML-Seldon Integration:

- **Seldon Deployment**: Configured in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`
  - Uses `SeldonDeploymentConfig` and `seldon_model_deployer_step` for Kubernetes deployment
  - Sets resource constraints and replicas for scalability
  - Automatically configures endpoints for API access

## 6. Front-end Client

The AUIChat project features a modern React-based frontend that provides an interactive chat experience:

### React Application:

- **UI Code**: Located in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI/auichat/`
- **Key Components**:
  - `ChatInterface.jsx`: Implements the chat interaction UI
  - `ThreeModel.jsx`: Creates a 3D visualization using Three.js
  - `ThemeProvider.jsx`: Provides dark/light mode themes

### Deployment Script:

- **UI Deployment**: `/home/barneh/Rag-Based-LLM_AUIChat/deploy_ui.sh`
  - Command-line script for deploying the UI to cloud providers
  - Supports both Firebase Hosting and Google Cloud Run deployment
  - Configurable via command-line arguments

The UI can be built and deployed using commands like:
```bash
./deploy_ui.sh --type firebase  # Deploy to Firebase Hosting
./deploy_ui.sh --type cloudrun --region us-central1  # Deploy to Cloud Run
```

## 7. Packaging and Containerization

The project has limited explicit containerization files, but the deployment pipelines handle containerization implicitly:

### Deployment Pipelines:

- **Cloud Run Deployment**: `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`
  - The `auichat_cloudrun_rag_pipeline()` function handles containerization for Cloud Run
  - The `deploy_cloudrun_rag_service()` function packages application components

- **CI/CD Configuration**: `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/workflows/data-sync.yml`
  - GitHub Actions workflow that handles deployment
  - Automates containerization and deployment steps

The containerization happens implicitly as part of the Cloud Run deployment process, which packages the application automatically.

## 8. ML Service Deployment

The project offers multiple deployment options, all orchestrated through ZenML pipelines:

### Deployment Options:

1. **Seldon Core Deployment**:
   - Pipeline: `auichat_seldon_deployment_pipeline()`
   - Purpose: Kubernetes-based deployment for scalable ML serving
   - Code: Lines 54-100 in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`

2. **Google Vertex AI Deployment**:
   - Pipeline: `auichat_vertex_deployment_pipeline()`
   - Purpose: Managed ML serving using Google's Vertex AI platform
   - Code: Lines 105-188 in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`

3. **Google Cloud Run Deployment**:
   - Pipeline: `auichat_cloudrun_rag_pipeline()`
   - Purpose: Serverless container deployment for the RAG service
   - Code: Lines 218-258 in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`

4. **Firebase UI Deployment**:
   - Pipeline: `auichat_firebase_ui_pipeline()`
   - Purpose: Static hosting for the React frontend
   - Code: Lines 282-302 in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`

5. **Full Deployment Pipeline**:
   - Pipeline: `auichat_full_deployment_pipeline()`
   - Purpose: End-to-end deployment of both backend and frontend components
   - Code: Lines 329-362 in `/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/main.py`

### Execution Method:

These deployment pipelines are executed using a simple command-line interface defined in the main module:

```python
if __name__ == "__main__":
    # Process environment variable for deployment type
    deployment_type = os.environ.get("AUICHAT_DEPLOYMENT", "").lower()
    
    # Process command line arguments
    if len(sys.argv) > 1:
        deployment_type = sys.argv[1].lower()
    
    # Activate appropriate stack based on deployment type
    if deployment_type in ["seldon", "vertex", "cloudrun", "firebase-ui", "cloudrun-ui", "full"]:
        activate_gcp_stack()
        
    # Run the appropriate pipeline based on deployment type
    if deployment_type == "seldon":
        print("🚀 Running pipeline with Seldon deployment...")
        auichat_seldon_deployment_pipeline()
    # ... additional deployment options
```

## 9. Cloud Integration

The project leverages several Google Cloud services for deployment and operation:

- **GCP Stack**: Activation via `activate_gcp_stack()` function in `main.py`
- **Google Cloud Storage**: For model artifact storage
- **Google Cloud Run**: For serverless container deployment
- **Vertex AI**: For managed model serving
- **Firebase**: For UI hosting

Each integration is configured and orchestrated through ZenML pipelines, providing a seamless deployment experience.

## 10. Conclusion

The ML productionization phase (Milestone 5) has successfully transformed the AUIChat system from a development project into a production-ready application with multiple deployment options. The use of ZenML pipelines ensures reproducibility and consistency across environments, while the modular architecture allows for flexible deployment targets.

Key achievements include:

1. Development of a responsive React UI for chat interaction
2. Integration of MLflow for model versioning and serving
3. Multiple cloud deployment options (Seldon, Vertex AI, Cloud Run)
4. Streamlined deployment through simple scripts and commands
5. A complete end-to-end pipeline from data preparation to UI deployment

These components collectively enable the AUIChat system to be deployed and scaled in various cloud environments, providing a production-quality retrieval-augmented generation chatbot for university information.