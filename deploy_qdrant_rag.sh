#!/bin/bash
# Script to deploy the Qdrant-connected AUIChat RAG model to Cloud Run

set -e

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
cp /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/preprocessed_nodes.pkl "${TMP_DIR}/"
cp /home/barneh/Rag-Based-LLM_AUIChat/improved_rag_app_qdrant.py "${TMP_DIR}/app.py"

# Create Dockerfile
cat > "${TMP_DIR}/Dockerfile" << 'EOF'
# Use Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PORT=8080

# Command to run the application
CMD ["python", "app.py"]
EOF

# Create enhanced requirements.txt with Qdrant client and embedding model
cat > "${TMP_DIR}/requirements.txt" << 'EOF'
flask>=2.0.0
numpy>=1.22.0
scikit-learn>=1.0.0
qdrant-client>=1.4.0
llama-index>=0.8.0
sentence-transformers>=2.2.2
torch>=2.0.0
transformers>=4.30.0
EOF

# Build and deploy to Cloud Run
echo "🔨 Building container image"
gcloud builds submit --tag "${IMAGE_NAME}" "${TMP_DIR}"

echo "🚀 Deploying to Cloud Run"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300s \
  --allow-unauthenticated

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
echo "python /home/barneh/Rag-Based-LLM_AUIChat/test_rag.py --url ${SERVICE_URL} --debug"

# Save deployment info
cat > /home/barneh/Rag-Based-LLM_AUIChat/cloudrun_qdrant_info.json << EOF
{
  "service_name": "${SERVICE_NAME}",
  "service_url": "${SERVICE_URL}",
  "project_id": "${PROJECT_ID}",
  "region": "${REGION}",
  "image_name": "${IMAGE_NAME}"
}
EOF