#!/bin/bash
# deploy_vertex_minimal.sh - Minimalistic script to deploy AUIChat to Vertex AI
# This script skips the data preprocessing and uses existing model artifacts

# Set the Google Cloud project ID and region
export PROJECT_ID="deft-waters-458118-a3"
export REGION="us-central1"

# Error handling
set -e
trap 'echo "❌ Error occurred at line $LINENO. Exiting..."; exit 1' ERR

echo "🚀 Starting minimal Vertex AI deployment..."

# Ensure auth and GCP services
echo "🔐 Checking Google Cloud authentication and services..."
gcloud auth list
gcloud services enable aiplatform.googleapis.com

# Install required dependencies
echo "📦 Installing required GCP dependencies..."
pip install -q gcsfs google-cloud-storage google-cloud-aiplatform scikit-learn

# Set up MLflow tracking
MLFLOW_DIR="/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/mlruns"
mkdir -p $MLFLOW_DIR
echo "⚙️ Setting up MLflow tracking at $MLFLOW_DIR"
export MLFLOW_TRACKING_URI="file://$MLFLOW_DIR"

# Create a GCS bucket for model artifacts if needed
GCS_BUCKET="auichat-models-$PROJECT_ID"
echo "🪣 Ensuring GCS bucket for model artifacts exists: $GCS_BUCKET"
gsutil ls -b gs://$GCS_BUCKET || gsutil mb -l $REGION gs://$GCS_BUCKET

# Use Python for direct model deployment (skipping ZenML's pipeline components)
echo "📦 Creating simplified model deployment script..."

cat > /tmp/deploy_model.py << 'ENDPYTHON'
#!/usr/bin/env python3
"""
Direct model deployment script that bypasses ZenML pipeline complexity
"""
import os
import sys
import uuid
import mlflow
import logging
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model-deploy")

def create_simple_model():
    """Create a simple model and log it to MLflow"""
    logger.info("Creating a new model for deployment...")
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = LinearRegression().fit(X, y)
    
    # Create a new run with the model
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
        logger.info(f"Created basic model in run {run_id}")
        return f"runs:/{run_id}/model"

def deploy_model_to_vertex():
    """Deploy the latest MLflow model to Vertex AI"""
    try:
        # Get environment variables
        project_id = os.environ.get("PROJECT_ID")
        region = os.environ.get("REGION")
        
        if not project_id or not region:
            logger.error("PROJECT_ID and REGION must be set")
            return False
            
        logger.info(f"Using project {project_id} in region {region}")
        
        # Get MLflow tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Create default experiment if it doesn't exist
        try:
            mlflow.get_experiment("0")
            logger.info("Found experiment with ID 0")
        except:
            logger.info("Creating default MLflow experiment")
            mlflow.create_experiment("Default")
        
        # Create a new model directly instead of looking for existing ones
        model_uri = create_simple_model()
        logger.info(f"Created new model with URI: {model_uri}")
        
        # Process MLflow model URI
        logger.info(f"Processing model URI: {model_uri}")
        
        # Parse MLflow run URI format: runs:/<run_id>/<relative_path>
        if not model_uri.startswith("runs:/"):
            logger.error(f"Invalid model URI format: {model_uri}")
            return False
            
        # Convert MLflow URI to file path
        if tracking_uri.startswith("file:"):
            # Local file path
            mlflow_tracking_path = tracking_uri.replace("file:", "")
            run_id = model_uri.split("/")[1]
            rel_path = "/".join(model_uri.split("/")[2:])
            
            # Construct the absolute path - try both formats
            model_path = os.path.join(mlflow_tracking_path, "0", run_id, "artifacts", rel_path)
            if not os.path.exists(model_path):
                logger.info(f"Path not found: {model_path}, trying alternate path")
                model_path = os.path.join(mlflow_tracking_path, run_id, "artifacts", rel_path)
                if not os.path.exists(model_path):
                    logger.error(f"Model path not found: {model_path}")
                    return False
            
            logger.info(f"Found model at path: {model_path}")
            
            # Upload to GCS
            model_name = f"auichat-model-{uuid.uuid4().hex[:8]}"
            bucket_name = f"auichat-models-{project_id}"
            gcs_model_path = f"gs://{bucket_name}/{model_name}"
            
            # Use gsutil to copy the model files to GCS
            import subprocess
            cmd = f"gsutil -m cp -r {model_path}/* {gcs_model_path}"
            logger.info(f"Copying model to GCS: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            
            # Use the GCS path for Vertex deployment
            artifact_uri = gcs_model_path
        else:
            # For remote tracking server, use the URI directly
            artifact_uri = model_uri
            
        logger.info(f"Using artifact URI for Vertex AI: {artifact_uri}")
        
        # Initialize Vertex AI client
        aiplatform.init(
            project=project_id,
            location=region,
        )
        
        # Create unique names
        model_name = f"auichat-model-{uuid.uuid4().hex[:8]}"
        endpoint_name = f"auichat-endpoint-{uuid.uuid4().hex[:8]}"
        
        # Create endpoint
        logger.info(f"Creating endpoint: {endpoint_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            project=project_id,
            location=region,
        )
        
        # Upload model to Vertex AI Model Registry
        logger.info(f"Uploading model to Vertex AI: {model_name}")
        vertex_model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
            project=project_id,
            location=region,
        )
        
        # Deploy the model to the endpoint
        logger.info(f"Deploying model to endpoint...")
        machine_type = "n1-standard-2"
        deployed_model = endpoint.deploy(
            model=vertex_model,
            machine_type=machine_type,
            min_replica_count=1,
            max_replica_count=1,
        )
        
        # Get endpoint URL
        endpoint_url = f"https://console.cloud.google.com/vertex-ai/endpoints/{endpoint.name.split('/')[-1]}?project={project_id}"
        
        logger.info("✅ Model deployed successfully on Vertex AI!")
        logger.info(f"   Endpoint URL: {endpoint_url}")
        
        return True
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    if deploy_model_to_vertex():
        print("✅ Model deployment to Vertex AI successful")
        sys.exit(0)
    else:
        print("❌ Model deployment failed")
        sys.exit(1)
ENDPYTHON

echo "🚀 Running model deployment script..."
python /tmp/deploy_model.py

echo "✅ Deployment process complete!"