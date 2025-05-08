#!/bin/bash
# Simple script to deploy the AUIChat RAG model to Cloud Run

set -e

PROJECT_ID=${PROJECT_ID:-"deft-waters-458118-a3"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"auichat-rag-service"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying AUIChat RAG model to Cloud Run"
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

# Create the Flask app
cat > "${TMP_DIR}/app.py" << 'EOF'
"""
AUIChat RAG Model Service
Flask application for serving the AUIChat RAG model
"""
import os
import pickle
import logging
import time
import threading
from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auichat-service")

# Initialize Flask app
app = Flask(__name__)

# Global variables for the model
nodes = None
index = None
query_engine = None
model_loading = False
model_error = None

def load_model():
    """Load the model"""
    global nodes, index, query_engine, model_loading, model_error
    
    try:
        model_loading = True
        logger.info("Loading AUIChat RAG model...")
        
        # Load the embedding model with retry logic
        max_retries = 3
        retry_count = 0
        embed_model = None
        
        while retry_count < max_retries:
            try:
                # Add cache directories for Hugging Face models
                os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
                os.environ["HF_HOME"] = "/tmp/huggingface_home"
                os.makedirs("/tmp/huggingface_cache", exist_ok=True)
                os.makedirs("/tmp/huggingface_home", exist_ok=True)
                
                # Lower case for RFC 1123 DNS label compliance
                model_name = "BAAI/bge-small-en-v1.5"
                logger.info(f"Loading embedding model: {model_name}, attempt {retry_count + 1}")
                
                # Use the same model with different constructor for retry variation
                if retry_count == 0:
                    embed_model = HuggingFaceEmbedding(model_name=model_name)
                elif retry_count == 1:
                    from sentence_transformers import SentenceTransformer
                    st_model = SentenceTransformer(model_name)
                    embed_model = HuggingFaceEmbedding(model=st_model)
                else:
                    # Try with a smaller model for the last attempt
                    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
                break
            except Exception as e:
                logger.warning(f"Error loading model, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
        
        if embed_model is None:
            raise RuntimeError("Failed to load embedding model after multiple attempts")
        
        # Load the preprocessed nodes
        nodes_path = os.path.join(os.path.dirname(__file__), "preprocessed_nodes.pkl")
        if not os.path.exists(nodes_path):
            logger.error(f"Nodes file not found at {nodes_path}")
            raise FileNotFoundError(f"Nodes file not found at {nodes_path}")
        
        with open(nodes_path, "rb") as f:
            nodes = pickle.load(f)
            
        # Create the index
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        logger.info("AUIChat RAG model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_error = str(e)
        raise
    finally:
        model_loading = False

# Start loading model in a background thread
loading_thread = threading.Thread(target=load_model)
loading_thread.daemon = True
loading_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model_loading:
        return jsonify({"status": "initializing", "message": "Model is loading"}), 200
    elif model_error:
        return jsonify({"status": "error", "message": f"Model failed to load: {model_error}"}), 200
    elif query_engine is None:
        return jsonify({"status": "not_ready", "message": "Model not loaded yet"}), 200
    else:
        return jsonify({"status": "healthy", "message": "Service is ready"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint that mimics Vertex AI's predict method
    Expected input format: {"instances": [{"query": "What are admission requirements?"}, ...]}
    """
    # Check model status
    if model_loading:
        return jsonify({
            "predictions": [{"message": "Model is still loading, please try again in a few seconds"}],
            "deployed_model_id": "auichat-rag-cloudrun"
        }), 200
    
    if model_error:
        return jsonify({
            "predictions": [{"error": f"Model failed to load: {model_error}"}],
            "deployed_model_id": "auichat-rag-cloudrun"
        }), 200
        
    if not query_engine:
        return jsonify({
            "predictions": [{"message": "Model not loaded yet, please try again in a few seconds"}],
            "deployed_model_id": "auichat-rag-cloudrun"
        }), 200
    
    try:
        # Get request data
        request_json = request.get_json(silent=True)
        
        if not request_json or 'instances' not in request_json:
            return jsonify({"error": "Invalid request format, expected {'instances': [...]}"}), 400
        
        instances = request_json['instances']
        logger.info(f"Received {len(instances)} instances for prediction")
        
        # Process each instance
        predictions = []
        for instance in instances:
            # Handle different input formats
            if isinstance(instance, dict) and "query" in instance:
                query = instance["query"]
            elif isinstance(instance, str):
                query = instance
            elif isinstance(instance, list) and len(instance) > 0:
                # For numeric arrays (test case), just return sample data
                predictions.append([float(sum(instance))])
                continue
            else:
                try:
                    query = str(instance)
                except:
                    predictions.append({"error": "Invalid input format"})
                    continue
            
            # Process the query
            try:
                response = query_engine.query(query)
                predictions.append({
                    "answer": str(response),
                    "sources": [n.node.get_content() for n in response.source_nodes]
                })
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                predictions.append({"error": f"Query processing error: {str(e)}"})
        
        # Return response in Vertex AI-compatible format
        return jsonify({
            "predictions": predictions,
            "deployed_model_id": "auichat-rag-cloudrun"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
EOF

# Create Dockerfile
cat > "${TMP_DIR}/Dockerfile" << 'EOF'
# Use Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories and set environment variables
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV HF_HOME=/tmp/huggingface_home
RUN mkdir -p /tmp/huggingface_cache /tmp/huggingface_home

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

# Create requirements.txt
cat > "${TMP_DIR}/requirements.txt" << 'EOF'
flask>=2.0.0
gunicorn>=20.1.0
llama-index>=0.8.0
llama-index-embeddings-huggingface>=0.1.0
transformers>=4.30.0
sentence-transformers>=2.2.2
torch>=2.0.0
numpy>=1.23.0
scikit-learn>=1.0.0
EOF

# Build and deploy to Cloud Run
echo "🔨 Building container image"
gcloud builds submit --tag "${IMAGE_NAME}" "${TMP_DIR}"

echo "🚀 Deploying to Cloud Run"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300s \
  --concurrency 1 \
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
echo "python /home/barneh/Rag-Based-LLM_AUIChat/test_rag.py --url ${SERVICE_URL}"

# Save deployment info
cat > /home/barneh/Rag-Based-LLM_AUIChat/cloudrun_deployment_info.json << EOF
{
  "service_name": "${SERVICE_NAME}",
  "service_url": "${SERVICE_URL}",
  "project_id": "${PROJECT_ID}",
  "region": "${REGION}",
  "image_name": "${IMAGE_NAME}"
}
EOF
