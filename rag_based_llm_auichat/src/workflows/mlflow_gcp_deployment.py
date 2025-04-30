"""
MLflow model deployer step for GCP-backed Seldon Core deployment
"""
from zenml import step
from typing import Dict, Any
import mlflow
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def prepare_mlflow_deployment_config(model_uri: str) -> Dict[str, Any]:
    """
    Prepares configuration for an MLflow model to be deployed to Seldon Core on GKE.
    
    This step creates the proper configuration to deploy an MLflow model 
    with Seldon Core and returns it.
    
    Args:
        model_uri: The MLflow model URI to deploy
        
    Returns:
        A dictionary with deployment configuration for the seldon_model_deployer_step
    """
    # Extract the model name from the URI
    model_name = model_uri.split('/')[-1].replace('_', '-').lower()
    
    # Make sure the model name is DNS-1123 compatible for Kubernetes
    if not model_name[0].isalpha():
        model_name = 'model-' + model_name
    
    logger.info(f"Preparing deployment config for model URI: {model_uri}")
    logger.info(f"Using Kubernetes-compatible model name: {model_name}")
    
    # Create configuration for the MLflow based Seldon deployment
    config = {
        "model_uri": model_uri,
        "model_name": model_name,
        "implementation": "MLFLOW_SERVER",
        "parameters": [
            {
                "name": "model_uri",
                "value": model_uri,
                "type": "STRING"
            }
        ],
        "resources": {
            "requests": {"cpu": "100m", "memory": "200Mi"},
            "limits": {"cpu": "200m", "memory": "500Mi"}
        },
        "replicas": 1
    }
    
    logger.info(f"Deployment configuration prepared: {config}")
    return config