"""
Cloud Run UI deployment step for AUIChat
Provides deployment to Google Cloud Run for the React UI
"""
from zenml import step
from zenml.logger import get_logger
import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

logger = get_logger(__name__)

@step
def deploy_ui_to_cloudrun(
    project_id: str = None, 
    region: str = "europe-west3",
    service_name: str = "auichat-ui"
) -> str:
    """
    Deploys the UI to Google Cloud Run.
    
    Args:
        project_id: Google Cloud project ID (optional, will use gcloud default if not provided)
        region: Google Cloud region to deploy to
        service_name: Name for the Cloud Run service
        
    Returns:
        The URL of the deployed Cloud Run service
    """
    logger.info("Starting Google Cloud Run UI deployment process")
    
    # Get current directory and project root
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent.parent
    ui_dir = os.path.join(project_root, "m5_frontend_client", "ui")
    
    # Check if UI directory exists
    if not os.path.exists(ui_dir):
        raise FileNotFoundError(f"UI directory not found at {ui_dir}")
    
    # Use provided project ID or get from gcloud
    if not project_id:
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                check=True
            )
            project_id = result.stdout.strip()
            logger.info(f"Using default GCP project: {project_id}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get default GCP project: {e}")
    
    # Create a temporary directory for Docker and deployment files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy UI files to temp directory
        logger.info(f"Copying UI files from {ui_dir} to {temp_dir}")
        shutil.copytree(ui_dir, os.path.join(temp_dir, "ui"), dirs_exist_ok=True)
        
        # Create Dockerfile
        dockerfile = """
FROM node:16-alpine as builder

# Set working directory
WORKDIR /app

# Copy UI files
COPY ui/ ./

# Install dependencies (if package.json exists)
RUN if [ -f package.json ]; then npm install; fi

# If no build script, just use the files as-is
RUN if [ -f package.json ] && grep -q "build" package.json; then \\
        npm run build; \\
    else \\
        mkdir -p build && cp -R * build/ 2>/dev/null || : ; \\
    fi

# Production stage with Nginx
FROM nginx:alpine

# Copy built app to nginx
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
RUN echo 'server { \\
    listen 8080; \\
    location / { \\
        root /usr/share/nginx/html; \\
        index index.html; \\
        try_files $uri $uri/ /index.html; \\
    } \\
}' > /etc/nginx/conf.d/default.conf

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
"""
        
        with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile)
        
        # Build the Docker image
        image_name = f"gcr.io/{project_id}/{service_name}:latest"
        logger.info(f"Building Docker image: {image_name}")
        
        try:
            subprocess.run(
                ["docker", "build", "-t", image_name, "."],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e.stderr}")
            raise RuntimeError(f"Docker build failed: {e.stderr}")
        
        # Push the image to Google Container Registry
        logger.info(f"Pushing image to Google Container Registry: {image_name}")
        try:
            # Configure docker to use gcloud credentials
            subprocess.run(
                ["gcloud", "auth", "configure-docker", "--quiet"],
                check=True,
                capture_output=True
            )
            
            # Push the image
            subprocess.run(
                ["docker", "push", image_name],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push Docker image: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            raise RuntimeError(f"Failed to push Docker image: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        
        # Deploy to Cloud Run
        logger.info(f"Deploying to Cloud Run: {service_name} in {region}")
        try:
            deploy_cmd = [
                "gcloud", "run", "deploy", service_name,
                "--image", image_name,
                "--platform", "managed",
                "--region", region,
                "--project", project_id,
                "--allow-unauthenticated",
                "--port", "8080"
            ]
            
            deploy_result = subprocess.run(
                deploy_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Extract service URL from deployment output
            output = deploy_result.stdout
            service_url = None
            for line in output.split('\n'):
                if "Service URL:" in line:
                    service_url = line.split("Service URL:")[1].strip()
                    break
            
            if not service_url:
                # Get the URL directly from gcloud
                url_result = subprocess.run(
                    [
                        "gcloud", "run", "services", "describe", service_name,
                        "--platform", "managed",
                        "--region", region,
                        "--project", project_id,
                        "--format", "value(status.url)"
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                service_url = url_result.stdout.strip()
            
            # Save deployment info to a file
            deployment_info = {
                "project_id": project_id,
                "region": region,
                "service_name": service_name,
                "service_url": service_url,
                "deployment_type": "cloudrun",
                "timestamp": subprocess.run(
                    ["date", "-u"], capture_output=True, text=True
                ).stdout.strip()
            }
            
            info_file = os.path.join(project_root.parent, "cloudrun_ui_info.json")
            with open(info_file, "w") as f:
                json.dump(deployment_info, f, indent=2)
            
            logger.info(f"UI successfully deployed to Cloud Run: {service_url}")
            logger.info(f"Deployment info saved to {info_file}")
            
            return service_url
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Cloud Run deployment failed: {e.stderr if hasattr(e, 'stderr') else str(e)}")
            raise RuntimeError(f"Cloud Run deployment failed: {e.stderr if hasattr(e, 'stderr') else str(e)}")