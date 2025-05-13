#!/bin/bash
# Script to set up scheduled RAG evaluation using Vertex AI and Cloud Scheduler

set -e  # Stop on any error

# Configuration
PROJECT_ID=${PROJECT_ID:-$(gcloud config get-value project)}
RAG_ENDPOINT=${RAG_ENDPOINT:-"https://auichat-rag-qdrant-448245131663.us-central1.run.app/predict"}
REGION=${REGION:-"us-central1"}
BUCKET_NAME=${BUCKET_NAME:-"auichat-rag-metrics"}
SCHEDULER_JOB_NAME="rag-vertex-evaluation"
SERVICE_ACCOUNT_NAME="rag-evaluation-sa"

echo "Setting up Vertex AI evaluation for RAG system..."
echo "Project ID: $PROJECT_ID"
echo "RAG Endpoint: $RAG_ENDPOINT"
echo "Region: $REGION"

# Install dependencies if needed
if ! python -c "import google.cloud.aiplatform" &>/dev/null; then
    echo "Installing dependencies..."
    pip install -r vertex_ai_requirements.txt
fi

# Ensure the bucket exists
echo "Checking GCS bucket..."
if ! gsutil ls -b gs://${BUCKET_NAME} &>/dev/null; then
    echo "Creating bucket: ${BUCKET_NAME}..."
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
else
    echo "Bucket exists: ${BUCKET_NAME}"
fi

# Create service account for automation if it doesn't exist
echo "Setting up service account..."
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create ${SERVICE_ACCOUNT_NAME} \
        --description="Service account for RAG evaluation" \
        --display-name="RAG Evaluation Service Account"
    
    # Grant necessary permissions
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/aiplatform.user"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
        --role="roles/storage.objectAdmin"
        
    echo "Service account created and permissions granted"
else
    echo "Service account already exists"
fi

# Create Cloud Function to run the evaluation
echo "Creating Cloud Function for evaluation..."

# Create a temporary directory for the Cloud Function code
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Create the Cloud Function code
cat > ${TEMP_DIR}/main.py << 'EOF'
import os
import sys
import json
import functions_framework
from google.cloud import storage
from datetime import datetime

# Add ML6 directory to the path
sys.path.append('/workspace')

# Import the evaluation module
from ML6.vertex_ai_evaluation import run_vertex_evaluation

@functions_framework.http
def evaluate_rag(request):
    """
    HTTP Cloud Function to evaluate RAG using Vertex AI
    """
    # Get configuration from environment variables
    project_id = os.environ.get('PROJECT_ID')
    region = os.environ.get('REGION', 'us-central1')
    endpoint_url = os.environ.get('RAG_ENDPOINT')
    bucket_name = os.environ.get('BUCKET_NAME', 'auichat-rag-metrics')
    
    print(f"Starting evaluation of RAG endpoint: {endpoint_url}")
    
    # Run the evaluation
    results = run_vertex_evaluation(
        endpoint_url=endpoint_url,
        project_id=project_id,
        region=region
    )
    
    # Return the results
    return json.dumps({
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "results": results
    }), 200, {'Content-Type': 'application/json'}
EOF

# Create requirements.txt for the Cloud Function
cat > ${TEMP_DIR}/requirements.txt << EOF
google-cloud-aiplatform>=1.25.0
google-cloud-storage>=2.0.0
pandas>=1.3.0
requests>=2.27.0
functions-framework>=3.0.0
EOF

# Copy the evaluation module to the Cloud Function directory
mkdir -p ${TEMP_DIR}/ML6
cp -v vertex_ai_evaluation.py ${TEMP_DIR}/ML6/

# Deploy the Cloud Function
echo "Deploying Cloud Function..."
gcloud functions deploy rag-evaluation-function \
    --gen2 \
    --region=${REGION} \
    --runtime=python310 \
    --source=${TEMP_DIR} \
    --entry-point=evaluate_rag \
    --trigger-http \
    --no-allow-unauthenticated \
    --service-account=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars=PROJECT_ID=${PROJECT_ID},REGION=${REGION},RAG_ENDPOINT=${RAG_ENDPOINT},BUCKET_NAME=${BUCKET_NAME} \
    --timeout=540s \
    --memory=1024MB

# Clean up temporary directory
rm -rf ${TEMP_DIR}

# Get the Cloud Function URL
FUNCTION_URL=$(gcloud functions describe rag-evaluation-function --region=${REGION} --gen2 --format="value(serviceConfig.uri)")

# Set up Cloud Scheduler to run the evaluation daily
echo "Setting up Cloud Scheduler job..."
SCHEDULER_EXISTS=$(gcloud scheduler jobs list --filter="name~${SCHEDULER_JOB_NAME}" --format="value(name)" 2>/dev/null || echo "")

# Create or update the scheduler job
if [ -z "$SCHEDULER_EXISTS" ]; then
    gcloud scheduler jobs create http ${SCHEDULER_JOB_NAME} \
        --schedule="0 0 * * *" \
        --uri="${FUNCTION_URL}" \
        --oidc-service-account-email=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
        --oidc-token-audience="${FUNCTION_URL}" \
        --http-method=POST \
        --time-zone="America/New_York" \
        --description="Daily evaluation of RAG system using Vertex AI"
else
    gcloud scheduler jobs update http ${SCHEDULER_JOB_NAME} \
        --schedule="0 0 * * *" \
        --uri="${FUNCTION_URL}" \
        --oidc-service-account-email=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
        --oidc-token-audience="${FUNCTION_URL}" \
        --http-method=POST \
        --time-zone="America/New_York" \
        --description="Daily evaluation of RAG system using Vertex AI"
fi

echo ""
echo "===== Setup Complete ====="
echo "Cloud Function: rag-evaluation-function"
echo "Scheduler Job: ${SCHEDULER_JOB_NAME} (runs daily at midnight)"
echo "Results will be stored in: gs://${BUCKET_NAME}/"
echo ""
echo "To run an evaluation manually:"
echo "python run_vertex_evaluation.py --endpoint $RAG_ENDPOINT --project $PROJECT_ID --region $REGION"
echo "================================"
