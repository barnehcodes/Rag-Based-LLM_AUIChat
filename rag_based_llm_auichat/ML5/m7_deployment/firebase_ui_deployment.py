"""
Firebase UI deployment step for AUIChat
Provides deployment to Firebase Hosting for the React UI
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
def deploy_ui_to_firebase(project_id: str = None) -> str:
    """
    Deploys the UI to Firebase Hosting.
    
    Args:
        project_id: Google Cloud project ID (optional, will use gcloud default if not provided)
        
    Returns:
        The URL of the deployed Firebase Hosting site
    """
    logger.info("Starting Firebase UI deployment process")
    
    # Get current directory and project root
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent
    
    # Update UI directory path to correct location
    ui_dir = os.path.join(project_root, "src", "UI", "auichat")
    
    logger.info(f"Looking for UI directory at: {ui_dir}")
    
    # Check if UI directory exists
    if not os.path.exists(ui_dir):
        # Try to find UI directory by searching
        logger.warning(f"UI directory not found at {ui_dir}, searching for it...")
        for root, dirs, _ in os.walk(project_root):
            if "auichat" in dirs and "UI" in root.split(os.sep):
                ui_dir = os.path.join(root, "auichat")
                logger.info(f"Found UI directory at: {ui_dir}")
                break
        
        if not os.path.exists(ui_dir):
            raise FileNotFoundError(f"UI directory not found at {ui_dir} or anywhere in the project")
    
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
    
    # First, build the UI
    logger.info("Building the React UI")
    try:
        # Check if node_modules exists, if not run npm install
        if not os.path.exists(os.path.join(ui_dir, "node_modules")):
            logger.info("Installing npm dependencies...")
            subprocess.run(
                ["npm", "install"],
                cwd=ui_dir,
                check=True
            )
        
        # Build the UI
        subprocess.run(
            ["npm", "run", "build"],
            cwd=ui_dir,
            check=True
        )
        
        # The build output should be in the "dist" directory
        build_dir = os.path.join(ui_dir, "dist")
        if not os.path.exists(build_dir):
            raise FileNotFoundError(f"Build directory not found at {build_dir}")
        
        logger.info(f"UI built successfully, output in {build_dir}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build UI: {e}")
    
    # Create a temporary directory for Firebase config
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy built UI files to temp directory
        logger.info(f"Copying built UI files from {build_dir} to {temp_dir}")
        shutil.copytree(build_dir, os.path.join(temp_dir, "public"), dirs_exist_ok=True)
        
        # Create firebase.json configuration file
        firebase_config = {
            "hosting": {
                "public": "public",
                "ignore": [
                    "firebase.json",
                    "**/.*",
                    "**/node_modules/**"
                ],
                "rewrites": [
                    {
                        "source": "**",
                        "destination": "/index.html"
                    }
                ]
            }
        }
        
        with open(os.path.join(temp_dir, "firebase.json"), "w") as f:
            json.dump(firebase_config, f, indent=2)
        
        # Initialize Firebase project with predefined site ID
        site_id = "auichat-rag-app"  # Predefined site ID
        logger.info(f"Using predefined site ID: {site_id}")
        
        try:
            subprocess.run(
                ["firebase", "hosting:sites:create", site_id, "--project", project_id],
                cwd=temp_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Site ID '{site_id}' might already exist. Proceeding with deployment.")
        
        # Deploy to Firebase Hosting
        logger.info("Deploying to Firebase Hosting (with debug output)...")
        deploy_cmd = [
            "firebase", "deploy",
            "--only", f"hosting:{site_id}",
            "--project", project_id,
            "--debug"  # Added debug flag for verbose output
        ]
        logger.info(f"Running command: {' '.join(deploy_cmd)}")
        
        # Run the command and stream its output
        process = subprocess.Popen(deploy_cmd, cwd=temp_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Stream stdout
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(f"[Firebase CLI STDOUT] {line.strip()}")
        
        # Stream stderr
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                logger.error(f"[Firebase CLI STDERR] {line.strip()}")

        process.wait() # Wait for the command to complete
        
        # Check return code
        if process.returncode != 0:
            # Error message will have been logged by stderr stream
            raise subprocess.CalledProcessError(
                process.returncode, 
                deploy_cmd, 
                output="See logs above for STDOUT", # stdout is now logged line by line
                stderr="See logs above for STDERR"  # stderr is now logged line by line
            )
        
        # Extract hosting URL from deployment output (this part might be tricky if output is only logged)
        # For now, we will rely on the fallback or assume the user sees it in the logs.
        # A more robust way would be to capture output to a string AND log it.
        # However, the primary request is to see the progress.
        
        # Fallback for hosting URL if not easily parsed from streamed logs
        hosting_url = f"https://{site_id}.web.app"
        logger.info(f"Assuming Hosting URL (please verify from Firebase CLI output above): {hosting_url}")
        
        # Save deployment info to a file
        deployment_info = {
            "project_id": project_id,
            "hosting_url": hosting_url,
            "deployment_type": "firebase",
            "timestamp": subprocess.run(
                ["date", "-u"], capture_output=True, text=True
            ).stdout.strip()
        }
        
        info_file = os.path.join(project_root.parent, "firebase_ui_info.json")
        with open(info_file, "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"UI successfully deployed to Firebase: {hosting_url}")
        logger.info(f"Deployment info saved to {info_file}")
        
        return hosting_url

if __name__ == "__main__":
    logger.info("Running Firebase UI deployment script directly...")
    
    current_project_id = None
    try:
        # Get project_id from gcloud config
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=True
        )
        current_project_id = result.stdout.strip()
        logger.info(f"Using GCP project ID from gcloud config: {current_project_id}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get default GCP project ID from gcloud: {e}")
        logger.error("Please ensure gcloud is configured or specify project_id manually.")
        # As a fallback, you might want to use a default or an environment variable
        # current_project_id = os.environ.get("PROJECT_ID", "your-default-project-id-if-any")
        # if not current_project_id or current_project_id == "your-default-project-id-if-any":
        #     logger.error("Project ID not found. Exiting.")
        #     exit(1) # Or handle as appropriate
    except FileNotFoundError:
        logger.error("gcloud command not found. Please ensure gcloud CLI is installed and in PATH.")
        # Fallback or exit
        # current_project_id = os.environ.get("PROJECT_ID", "your-default-project-id-if-any")
        # if not current_project_id or current_project_id == "your-default-project-id-if-any":
        #     logger.error("Project ID not found. Exiting.")
        #     exit(1)

    if not current_project_id:
        logger.error("Could not determine project_id. Please configure gcloud or set PROJECT_ID environment variable.")
    else:
        try:
            # Pass the fetched project_id to the function
            deployed_url = deploy_ui_to_firebase(project_id=current_project_id)
            if deployed_url:
                logger.info(f"✅ UI Deployment Successful. Hosting URL: {deployed_url}")
            else:
                logger.error("❌ UI Deployment failed. No URL returned.")
        except Exception as e:
            logger.error(f"❌ UI Deployment script failed with error: {e}")
            import traceback
            logger.error(traceback.format_exc())