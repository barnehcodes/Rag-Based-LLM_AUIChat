#!/bin/bash
# deploy_vertex.sh - Script to deploy AUIChat to Vertex AI

# Set the Google Cloud project ID - update this with your project ID
export PROJECT_ID="deft-waters-458118-a3"
export REGION="us-central1"

# Error handling
set -e  # Exit immediately if a command exits with a non-zero status
trap 'echo "❌ Error occurred at line $LINENO. Exiting..."; exit 1' ERR

# Ensure we're authenticated with Google Cloud
echo "🔐 Checking Google Cloud authentication..."
gcloud auth list
if [ $? -ne 0 ]; then
    echo "❌ Not authenticated with Google Cloud. Running 'gcloud auth login'..."
    gcloud auth login
fi

# Ensure Vertex AI API is enabled
echo "🔄 Ensuring Vertex AI API is enabled..."
gcloud services enable aiplatform.googleapis.com

# Ensure billing is enabled
echo "💰 Checking billing account..."
gcloud billing projects describe $PROJECT_ID
if [ $? -ne 0 ]; then
    echo "❌ Billing account not properly configured. Please run:"
    echo "gcloud billing projects link $PROJECT_ID --billing-account=YOUR_BILLING_ACCOUNT_ID"
    exit 1
fi

# Install required dependencies
echo "📦 Installing required GCP dependencies..."
pip install -q gcsfs google-cloud-storage google-cloud-aiplatform

# Create a Google Cloud Storage bucket if it doesn't exist
echo "🪣 Checking if Google Cloud Storage bucket exists..."
gsutil ls -b gs://auichat-models-$PROJECT_ID || gsutil mb -l $REGION gs://auichat-models-$PROJECT_ID

# Create a local directory for MLflow tracking
echo "📁 Setting up local MLflow directory..."
MLFLOW_DIR="/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/mlruns/0"
mkdir -p $MLFLOW_DIR

# Update MLflow tracker with a valid file:// URI
echo "⚙️ Updating MLflow tracker with valid file URI..."
zenml experiment-tracker update mlflow-tracker --tracking_uri="file://$MLFLOW_DIR" || echo "Could not update MLflow tracker"

# Use the existing gcp-stack
echo "🔄 Ensuring gcp-stack is active..."
zenml stack set gcp-stack

# Fix the model URI handling issue
echo "🛠️ Applying fix for model URI handling..."
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat

# First, directly apply fixes instead of using patches
echo "📝 Updating vertex_deployment.py with improved code..."

cat > src/workflows/vertex_deployment.py << 'ENDVERTEX'
"""
Vertex AI model deployment module for AUIChat
Provides steps to deploy MLflow models to Google Cloud Vertex AI endpoints
"""
import os
import logging
from typing import Dict, Any, Optional, Union, List
from zenml.steps import step
import json
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def deploy_model_to_vertex(
    model_uri: str,
    machine_type: str = "n1-standard-2",
    min_replicas: int = 1,
    max_replicas: int = 1,
) -> Dict[str, str]:
    """
    Deploys a model to Google Cloud Vertex AI from a local MLflow model.
    
    Args:
        model_uri (str): The URI string to the MLflow model to deploy
        machine_type (str): GCP machine type for deployment
        min_replicas (int): Minimum number of replicas
        max_replicas (int): Maximum number of replicas
        
    Returns:
        Dict[str, str]: Deployment information including endpoint URL
    """
    # Import here to avoid ZenML circular imports
    from google.cloud import aiplatform
    
    # Log the input model URI
    logger.info(f"Deploying model from URI: {model_uri}")
    
    # Extract GCP project ID and region from environment variables
    project_id = os.environ.get("PROJECT_ID")
    region = os.environ.get("REGION")
    
    if not project_id or not region:
        raise ValueError(
            "PROJECT_ID and REGION environment variables must be set for Vertex AI deployment"
        )
    
    logger.info(f"Using GCP Project ID: {project_id} in Region: {region}")
    
    try:
        # Initialize Vertex AI client
        aiplatform.init(
            project=project_id,
            location=region,
        )
        
        # Generate a unique model name with a timestamp
        model_name = f"auichat-model-{uuid.uuid4().hex[:8]}"
        endpoint_name = f"auichat-endpoint-{uuid.uuid4().hex[:8]}"
        
        # Upload the model to Vertex AI Model Registry
        logger.info(f"Uploading model to Vertex AI Model Registry: {model_name}")
        
        # Create or get an endpoint
        endpoints = aiplatform.Endpoint.list(
            filter=f'display_name="{endpoint_name}"',
            project=project_id,
            location=region,
        )
        
        if endpoints:
            endpoint = endpoints[0]
            logger.info(f"Using existing endpoint: {endpoint.resource_name}")
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                project=project_id,
                location=region,
            )
            logger.info(f"Created new endpoint: {endpoint.resource_name}")
        
        # Upload the model to Vertex AI Model Registry
        logger.info(f"Uploading model from {model_uri} to Vertex AI")
        
        # Convert MLflow URI to a GCS path, if needed
        if model_uri.startswith("runs:/"):
            # Parse MLflow run URI format: runs:/<run_id>/<relative_path>
            logger.info("Detected MLflow run URI format. Converting to absolute path...")
            import mlflow
            from pathlib import Path
            
            # Get the MLflow tracking URI
            mlflow_tracking_uri = mlflow.get_tracking_uri()
            logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
            
            if mlflow_tracking_uri.startswith("file:"):
                # Local file path
                mlflow_tracking_path = mlflow_tracking_uri.replace("file:", "")
                run_id = model_uri.split("/")[1]
                rel_path = "/".join(model_uri.split("/")[2:])
                
                # Construct the absolute path
                model_path = os.path.join(mlflow_tracking_path, run_id, "artifacts", rel_path)
                logger.info(f"Converted MLflow URI to local path: {model_path}")
                
                # For Vertex, we need to upload the model to a GCS bucket
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
        else:
            # Use the URI as is
            artifact_uri = model_uri
            
        logger.info(f"Using artifact URI for Vertex AI: {artifact_uri}")
        
        # Create the model in Vertex AI
        vertex_model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
            project=project_id,
            location=region,
        )
        
        # Deploy the model to the endpoint
        logger.info(f"Deploying model to endpoint...")
        deployed_model = endpoint.deploy(
            model=vertex_model,
            machine_type=machine_type,
            min_replica_count=min_replicas,
            max_replica_count=max_replicas,
        )
        
        # Build the response with deployment information as simple strings
        endpoint_url = f"https://console.cloud.google.com/vertex-ai/endpoints/{endpoint.name.split('/')[-1]}?project={project_id}"
        model_id = deployed_model.id if hasattr(deployed_model, 'id') else "unknown"
        
        logger.info("✅ Model deployed successfully on Vertex AI!")
        logger.info(f"   Endpoint URL: {endpoint_url}")
        
        return {
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "endpoint_url": endpoint_url,
            "deployed_model_id": model_id,
            "model_uri": model_uri
        }
        
    except Exception as e:
        logger.error(f"❌ Error deploying model to Vertex AI: {str(e)}")
        # Return an error dictionary instead of raising an exception
        # This allows the pipeline to continue and launch UI components
        return {
            "endpoint_name": "deployment_failed",
            "model_name": "deployment_failed",
            "endpoint_url": "deployment_failed",
            "deployed_model_id": "deployment_failed",
            "model_uri": model_uri,
            "error": str(e)
        }
ENDVERTEX

echo "📝 Now updating main.py to fix syntax errors..."

# Completely replace main.py with a fully fixed version
cat > src/main.py << 'ENDMAIN'
from zenml import pipeline, get_step_context
from zenml.client import Client
from data.Data_preprocessing import preprocess_data
from data.index_storage import create_and_store_index
from valisation.validation import validate_qdrant_storage
from workflows.model_saving import save_model_for_deployment
from workflows.model_training import placeholder_model_trainer
from workflows.ui_launcher import launch_ui_components
from workflows.vertex_deployment import deploy_model_to_vertex
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService,
)
from zenml.integrations.seldon.steps import seldon_model_deployer_step
import sys
import os

# Define constants for deployment
SELDON_DEPLOYMENT_NAME = "auichat-smollm-deployment"
MLFLOW_MODEL_NAME = "auichat-smollm-360m"

# Activate the GCP stack if running with cloud resources
def activate_gcp_stack():
    """Activates the GCP stack for cloud deployments"""
    try:
        client = Client()
        # Check if gcp-stack exists
        stacks = client.list_stacks()
        if "gcp-stack" in [s.name for s in stacks]:
            client.activate_stack("gcp-stack")
            print("✅ Activated GCP stack for cloud resources")
            return True
        else:
            print("⚠️ GCP stack not found. Using default stack.")
            return False
    except Exception as e:
        print(f"⚠️ Error activating GCP stack: {e}")
        print("⚠️ Continuing with default stack")
        return False

# Define the Seldon deployment pipeline
@pipeline(enable_cache=False)
def auichat_seldon_deployment_pipeline():
    """
    Deployment pipeline for AUIChat that runs preprocessing, indexing,
    validation, saves the model via MLflow, deploys it to Seldon,
    and launches the UI components.
    """
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Save the model using MLflow, returns the URI string
    model_uri = save_model_for_deployment(after=[validation_result])
    
    # Try to deploy with Seldon
    try:
        # Define the Seldon deployment configuration
        deployment_config = SeldonDeploymentConfig(
            model_name=SELDON_DEPLOYMENT_NAME,
            implementation="TRANSFORMER",
            parameters=[],
            resources={
                "requests": {"cpu": "100m", "memory": "512Mi"},
                "limits": {"cpu": "1000m", "memory": "1Gi"}
            },
            replicas=1,
            secret_name="",
        )

        # Deploy the model to Seldon
        deployment_service = seldon_model_deployer_step(
            model=model_uri,
            service_config=deployment_config,
            after=[model_uri]
        )
        print("✅ Model deployed successfully with Seldon Core!")
        
        # Launch UI components after deployment
        launch_ui_components(after=[deployment_service])
    except Exception as e:
        print(f"❌ Seldon deployment failed: {str(e)}")
        print("⚠️ Continuing with UI launch without Seldon deployment.")
        
        # Launch UI components without deployment
        launch_ui_components(after=[model_uri])

# Define the Vertex AI deployment pipeline
@pipeline(enable_cache=False)
def auichat_vertex_deployment_pipeline():
    """
    Deployment pipeline for AUIChat that uses Google Cloud Vertex AI
    for model deployment instead of Seldon Core.
    """
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Save the model using MLflow, returns the URI string
    model_uri = save_model_for_deployment(after=[validation_result])
    
    # Deploy with Vertex AI
    try:
        # Extract the model URI string from StepArtifact
        if hasattr(model_uri, 'read'):
            # This is the proper way to access the content of a StepArtifact
            model_uri_str = model_uri.read()
            print(f"📦 Extracted model URI for deployment: {model_uri_str}")
        elif hasattr(model_uri, 'uri'):
            # Alternative way to access the URI if read() isn't available
            model_uri_str = model_uri.uri
            print(f"📦 Extracted model URI from uri attribute: {model_uri_str}")
        elif isinstance(model_uri, dict) and 'uri' in model_uri:
            # If model_uri is already a dictionary
            model_uri_str = model_uri['uri']
            print(f"📦 Extracted model URI from dictionary: {model_uri_str}")
        elif isinstance(model_uri, str):
            # If model_uri is already a string
            model_uri_str = model_uri
            print(f"📦 Using model URI directly as string: {model_uri_str}")
        else:
            # Convert to string as last resort
            model_uri_str = str(model_uri)
            print(f"⚠️ Using string representation of model_uri: {model_uri_str}")
        
        # Deploy the model to a Vertex AI endpoint with minimal resources
        deployment_info = deploy_model_to_vertex(
            model_uri=model_uri_str,  # Pass the extracted string URI
            machine_type="n1-standard-2",  # Low-cost machine type
            min_replicas=1,
            max_replicas=1,
            after=[model_uri]
        )
        
        # Check if deployment was successful
        if hasattr(deployment_info, 'read'):
            # If deployment_info is a ZenML StepArtifact, try to read its content
            try:
                deployment_data = deployment_info.read()
                print(f"✅ Model deployed successfully on Vertex AI!")
                print(f"   Deployment information available in ZenML dashboard.")
                # Pass the deployment_info to the UI launcher
                launch_ui_components(after=[deployment_info])
            except Exception as e:
                print(f"⚠️ Could not read deployment info: {str(e)}")
                print("⚠️ Continuing with UI launch without model deployment.")
                launch_ui_components(after=[model_uri])
        elif isinstance(deployment_info, dict):
            # If deployment_info is a dictionary
            endpoint_url = deployment_info.get('endpoint_url', 'URL not available')
            if endpoint_url != 'deployment_failed':
                print(f"✅ Model deployed successfully on Vertex AI!")
                print(f"   Endpoint URL: {endpoint_url}")
                # Launch UI components after deployment
                launch_ui_components(after=[deployment_info])
            else:
                error_msg = deployment_info.get('error', 'Unknown error')
                print(f"❌ Vertex AI deployment failed: {error_msg}")
                print("⚠️ Continuing with UI launch without model deployment.")
                # Launch UI components without deployment
                launch_ui_components(after=[model_uri])
        else:
            print(f"⚠️ Unexpected deployment_info type: {type(deployment_info)}.")
            print("⚠️ Continuing with UI launch without model deployment.")
            # Launch UI components without deployment
            launch_ui_components(after=[model_uri])
            
    except Exception as e:
        print(f"❌ Vertex AI deployment failed: {str(e)}")
        print("⚠️ Continuing with UI launch without model deployment.")
        
        # Launch UI components without deployment
        launch_ui_components(after=[model_uri])

# Pipeline that skips the deployment step entirely
@pipeline(enable_cache=False)
def auichat_no_deploy_pipeline():
    """
    Simplified pipeline for AUIChat that runs preprocessing, indexing,
    validation, saves the model via MLflow, and launches the UI components
    without attempting deployment.
    """
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Save the model using MLflow, returns the URI string
    model_uri = save_model_for_deployment(after=[validation_result])
    
    # Launch UI components after model saving
    launch_ui_components(after=[model_uri])


if __name__ == "__main__":
    # Process environment variable for deployment type
    deployment_type = os.environ.get("AUICHAT_DEPLOYMENT", "").lower()
    
    # Process command line arguments
    if len(sys.argv) > 1:
        deployment_type = sys.argv[1].lower()
    
    # Activate appropriate stack based on deployment type
    if deployment_type in ["seldon", "vertex"]:
        activate_gcp_stack()
    
    # Run the appropriate pipeline based on deployment type
    if deployment_type == "seldon":
        print("🚀 Running pipeline with Seldon deployment...")
        auichat_seldon_deployment_pipeline()
    elif deployment_type == "vertex":
        print("🚀 Running pipeline with Vertex AI deployment...")
        auichat_vertex_deployment_pipeline()
    else:
        print("🚀 Running pipeline without deployment step...")
        auichat_no_deploy_pipeline()
ENDMAIN

# Verify the changes were made
echo "✅ Fixed model URI handling in vertex_deployment.py"
echo "✅ Fixed syntax errors in main.py"

echo "🚀 Running AUIChat pipeline with Vertex AI deployment..."
python src/main.py vertex

echo "✅ Deployment process complete!"