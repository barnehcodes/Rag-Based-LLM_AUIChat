#!/bin/bash
# Script to deploy the Qdrant-connected AUIChat RAG model to Cloud Run

set -e

# Determine project root (parent of the script's directory)
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

PROJECT_ID=${PROJECT_ID:-"deft-waters-458118-a3"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"auichat-rag-qdrant"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying Qdrant-connected AUIChat RAG model to Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"

# Create a temporary directory
TMP_DIR=$(mktemp -d)
echo "📁 Created temporary directory: ${TMP_DIR}"

# Cleanup function
cleanup() {
  echo "🧹 Cleaning up temporary directory"
  rm -rf "${TMP_DIR}"
}

# Register cleanup on exit
trap cleanup EXIT

# Copy necessary files
echo "📂 Preparing application files"
cp "${PROJECT_ROOT}/rag_based_llm_auichat/preprocessed_nodes.pkl" "${TMP_DIR}/"
cp "${PROJECT_ROOT}/improved_rag_app_qdrant.py" "${TMP_DIR}/app.py"

# Create Dockerfile
cat > "${TMP_DIR}/Dockerfile" << 'EOF'
# Use Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up HuggingFace cache environment variables to be used during build and runtime
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the preload_models.py script
COPY preload_models.py .

# Run the script to download models into the cache layer of the image
# Pass the LLM_MODEL_NAME_FOR_PREBAKE as a build argument
ARG LLM_MODEL_NAME_FOR_PREBAKE_ARG=HuggingFaceTB/SmolLM-360M-Instruct
ENV LLM_MODEL_NAME_FOR_PREBAKE=${LLM_MODEL_NAME_FOR_PREBAKE_ARG}
RUN python preload_models.py

# Copy the rest of the application files
COPY . .

# Set environment variables for runtime (PORT is already standard)
# QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, LLM_MODEL_NAME will be set by Cloud Run

# Command to run the application
CMD [ "python", "app.py" ]
EOF

# Create a script to preload models (will be run inside Docker build)
cat > "${TMP_DIR}/preload_models.py" << 'EOF'
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preload_models")

# Cache directory is set by ENV in Dockerfile
cache_dir = os.environ.get("HF_HOME", "/app/huggingface_cache")
logger.info(f"Using HuggingFace cache directory: {cache_dir}")

# Embedding model
embed_model_name = os.environ.get("EMBEDDING_MODEL_NAME_FOR_PREBAKE", "BAAI/bge-small-en-v1.5") # Changed default and made configurable
try:
    logger.info(f"Downloading embedding model: {embed_model_name} to {cache_dir}")
    SentenceTransformer(embed_model_name, cache_folder=cache_dir)
    logger.info(f"Embedding model {embed_model_name} downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download embedding model {embed_model_name}: {e}", exc_info=True)
    raise

# LLM - Get model name from environment variable set by Docker ARG
llm_model_name = os.environ.get("LLM_MODEL_NAME_FOR_PREBAKE")
if not llm_model_name:
    logger.error("LLM_MODEL_NAME_FOR_PREBAKE environment variable not set for preloading.")
    raise ValueError("LLM model name for pre-baking not provided.")

try:
    logger.info(f"Downloading LLM tokenizer: {llm_model_name} to {cache_dir}")
    AutoTokenizer.from_pretrained(llm_model_name)
    logger.info(f"Downloading LLM model: {llm_model_name} to {cache_dir}")
    AutoModelForCausalLM.from_pretrained(llm_model_name)
    logger.info(f"LLM {llm_model_name} downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download LLM {llm_model_name}: {e}", exc_info=True)
    raise

logger.info("Model pre-loading script completed.")
EOF

# Create enhanced requirements.txt with Qdrant client and embedding model
cat > "${TMP_DIR}/requirements.txt" << 'EOF'
flask>=2.0.0
flask-cors>=4.0.0
numpy>=1.22.0
scikit-learn>=1.0.0
qdrant-client>=1.4.0
sentence-transformers>=2.2.2
torch>=2.0.0
transformers>=4.30.0
llama-index-core
llama-index-vector-stores-qdrant
llama-index-embeddings-huggingface
llama-index-llms-huggingface
EOF

# Create cloudbuild.yaml
cat > "${TMP_DIR}/cloudbuild.yaml" << 'EOF'
steps:
- name: 'gcr.io/cloud-builders/docker'
  args:
  - 'build'
  - '-t'
  - '${_IMAGE_NAME}'  # Substitution for image name
  - '--build-arg'
  - 'LLM_MODEL_NAME_FOR_PREBAKE_ARG=${_LLM_MODEL_NAME_FOR_PREBAKE_SUB}' # Substitution for the build arg
  - '--build-arg' # Added for embedding model
  - 'EMBEDDING_MODEL_NAME_FOR_PREBAKE_ARG=${_EMBEDDING_MODEL_NAME_FOR_PREBAKE_SUB}' # Added for embedding model
  - '.' # Build context (the current directory where Dockerfile is)
  id: 'Build Docker Image'

substitutions:
  _IMAGE_NAME: 'gcr.io/default-project/default-image' # Default, will be overridden
  _LLM_MODEL_NAME_FOR_PREBAKE_SUB: 'HuggingFaceTB/SmolLM-360M-Instruct' # Default, will be overridden
  _EMBEDDING_MODEL_NAME_FOR_PREBAKE_SUB: 'BAAI/bge-small-en-v1.5' # Added default for embedding model

# Optional: Increase the build timeout if model downloads are very slow
timeout: '3600s' # e.g., 1 hour, default is 10 minutes (600s)

images:
- '${_IMAGE_NAME}' # Specifies the image to be pushed to GCR upon successful build
EOF

# Build and deploy to Cloud Run
echo "🔨 Building container image using cloudbuild.yaml (this may take a while due to model downloads)..."
# Pass the LLM_MODEL_NAME and IMAGE_NAME as substitutions
LLM_FOR_PREBAKE=${LLM_MODEL_NAME:-HuggingFaceTB/SmolLM-360M-Instruct}
EMBEDDING_FOR_PREBAKE=${EMBEDDING_MODEL_NAME:-BAAI/bge-small-en-v1.5} # Added for embedding model

gcloud builds submit "${TMP_DIR}" \
  --config "${TMP_DIR}/cloudbuild.yaml" \
  --substitutions "_IMAGE_NAME=${IMAGE_NAME},_LLM_MODEL_NAME_FOR_PREBAKE_SUB=${LLM_FOR_PREBAKE},_EMBEDDING_MODEL_NAME_FOR_PREBAKE_SUB=${EMBEDDING_FOR_PREBAKE}" # Added embedding model substitution

echo "🚀 Deploying to Cloud Run"
# Retrieve QDRANT_URL and QDRANT_API_KEY from environment or use defaults
# IMPORTANT: Set these in your shell environment before running, or replace placeholders.
# Example: export QDRANT_URL="your_qdrant_url"
#          export QDRANT_API_KEY="your_qdrant_api_key"
QDRANT_URL_TO_USE=${QDRANT_URL:-"https://40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"}
QDRANT_API_KEY_TO_USE=${QDRANT_API_KEY:-"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM"}
QDRANT_COLLECTION_TO_USE=${QDRANT_COLLECTION:-"AUIChatVectoreCol-384"}

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --memory 4Gi \
  --cpu 2 \
  --timeout 600s \
  --cpu-boost \
  --min-instances 1 \
  --max-instances 5 \
  --allow-unauthenticated \
  --set-env-vars="QDRANT_URL=${QDRANT_URL_TO_USE}" \
  --set-env-vars="^##^QDRANT_API_KEY=${QDRANT_API_KEY_TO_USE}" \
  --set-env-vars="QDRANT_COLLECTION=${QDRANT_COLLECTION_TO_USE}" \
  --set-env-vars="LLM_MODEL_NAME=${LLM_MODEL_NAME:-HuggingFaceTB/SmolLM-360M-Instruct}" \
  --set-env-vars="EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME:-BAAI/bge-small-en-v1.5}" \
  --no-cpu-throttling

# Get the service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --format="value(status.url)")

echo "✅ Deployment successful!"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "To use this endpoint in your code, set the environment variable:"
echo "export AUICHAT_ENDPOINT_URL=${SERVICE_URL}"
echo ""
echo "To run tests:"
echo "python ${PROJECT_ROOT}/test_rag.py --url ${SERVICE_URL} --debug"

# Save deployment info to project root
cat > "${PROJECT_ROOT}/cloudrun_qdrant_info.json" << EOF
{
  "service_name": "${SERVICE_NAME}",
  "service_url": "${SERVICE_URL}",
  "project_id": "${PROJECT_ID}",
  "region": "${REGION}",
  "image_name": "${IMAGE_NAME}"
}
EOF
echo "Deployment info saved to ${PROJECT_ROOT}/cloudrun_qdrant_info.json"

