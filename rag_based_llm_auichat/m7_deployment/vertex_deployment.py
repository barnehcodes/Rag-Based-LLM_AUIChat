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
