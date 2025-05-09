from zenml import step
from zenml.logger import get_logger
import os
import subprocess
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = get_logger(__name__)

@step
def deploy_improved_rag_app_step(
    project_id: Optional[str] = None,
    region: str = "us-central1",
    service_name: str = "auichat-rag-app",
    qdrant_url_env: Optional[str] = None,
    qdrant_api_key_env: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploys the improved_rag_app_qdrant.py to Cloud Run using the deploy_rag_backend_cloudrun.sh script.
    
    Args:
        project_id: GCP project ID (uses default if None)
        region: GCP region for deployment
        service_name: Name of the Cloud Run service
        qdrant_url_env: Optional Qdrant URL environment variable
        qdrant_api_key_env: Optional Qdrant API key environment variable
        
    Returns:
        Dictionary with deployment information including service URL
    """
    logger.info(f"Deploying improved RAG app to Cloud Run: {service_name}")
    
    # Determine Project Root dynamically
    project_root = Path("/home/barneh/Rag-Based-LLM_AUIChat")
    script_path = project_root / "deployment_scripts" / "deploy_rag_backend_cloudrun.sh"
    
    if not script_path.exists():
        logger.error(f"Deployment script not found: {script_path}")
        raise FileNotFoundError(f"Deployment script not found: {script_path}")
    
    # Use provided project_id or get from environment variables
    resolved_project_id = project_id or os.environ.get("PROJECT_ID") or os.environ.get("GCP_PROJECT_ID")
    if not resolved_project_id:
        logger.error("GCP Project ID not provided and not found in environment variables.")
        raise ValueError("GCP Project ID is required for deployment.")
    
    try:
        # Prepare environment variables for the script
        env = os.environ.copy()
        env["PROJECT_ID"] = resolved_project_id
        env["REGION"] = region
        env["SERVICE_NAME"] = service_name
        
        # Pass through Qdrant details if provided
        if qdrant_url_env:
            env["QDRANT_URL"] = qdrant_url_env
        if qdrant_api_key_env:
            env["QDRANT_API_KEY"] = qdrant_api_key_env
        
        # Make sure the script is executable
        subprocess.run(["chmod", "+x", script_path], check=True)
        
        # Execute the deployment script
        logger.info(f"Executing deployment script: {script_path}")
        logger.info(f"With PROJECT_ID={resolved_project_id}, REGION={region}, SERVICE_NAME={service_name}")
        
        process = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=str(project_root)
        )
        
        logger.info("Deployment script executed successfully.")
        if process.stdout:
            logger.info(f"Script output: {process.stdout}")
        if process.stderr:
            logger.warning(f"Script stderr: {process.stderr}")
        
        # Check for deployment info file created by the script
        info_file_path = project_root / "cloudrun_qdrant_info.json"
        if info_file_path.exists():
            with open(info_file_path, 'r') as f:
                deployment_info = json.load(f)
            logger.info(f"Successfully read deployment info from {info_file_path}")
            
            # Ensure service_url is present in the returned info
            if "service_url" not in deployment_info:
                if "url" in deployment_info:
                    deployment_info["service_url"] = deployment_info["url"]
                else:
                    logger.warning("service_url not found in deployment info. Subsequent steps might fail.")
                    deployment_info["service_url"] = "URL_NOT_FOUND"
            
            # Add additional info
            deployment_info.setdefault("service_name", service_name)
            deployment_info.setdefault("project_id", resolved_project_id)
            deployment_info.setdefault("region", region)
            
            return deployment_info
        else:
            # If info file not found, try to get service URL from command output
            logger.warning(f"Deployment info file '{info_file_path}' not found.")
            service_url = None
            
            for line in process.stdout.splitlines():
                if "Service URL:" in line:
                    service_url = line.split("Service URL:")[1].strip()
                    break
                    
            if service_url:
                logger.info(f"Parsed service URL from output: {service_url}")
                return {
                    "service_name": service_name,
                    "service_url": service_url,
                    "project_id": resolved_project_id,
                    "region": region
                }
            else:
                logger.error("Could not determine service URL from script output.")
                return {
                    "service_name": service_name,
                    "service_url": "UNKNOWN",
                    "project_id": resolved_project_id,
                    "region": region
                }
                
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment script failed: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Cloud Run deployment failed: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error during deployment: {e}")
        raise