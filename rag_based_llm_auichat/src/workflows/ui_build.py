from zenml import step
from zenml.logger import get_logger
import os
import subprocess
from pathlib import Path

logger = get_logger(__name__)

@step
def build_ui_for_firebase_step() -> str:
    """
    Checks if the React UI build directory (dist) exists and runs npm build if not.
    
    Returns:
        String path to the dist directory
    """
    # Path to the React UI directory
    ui_dir = Path("/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI/auichat")
    dist_dir = ui_dir / "dist"
    
    logger.info(f"Checking if UI build directory exists: {dist_dir}")
    
    # Check if dist directory exists and is not empty
    if dist_dir.exists() and any(dist_dir.iterdir()):
        logger.info("UI build directory already exists and contains files. Skipping build.")
        return str(dist_dir)
    
    # Directory doesn't exist or is empty, need to build
    logger.info("UI build directory not found or empty. Running npm build...")
    
    # Ensure we're in the UI directory
    if not ui_dir.exists():
        raise FileNotFoundError(f"UI directory not found at: {ui_dir}")
    
    try:
        # Check if node_modules exists, if not run npm install first
        if not (ui_dir / "node_modules").exists():
            logger.info("node_modules not found. Running npm install first...")
            subprocess.run(
                ["npm", "install"],
                cwd=str(ui_dir),
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("npm install completed successfully")
        
        # Build the UI
        logger.info("Running npm run build...")
        build_process = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(ui_dir),
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("npm build completed successfully")
        
        # Verify that dist directory now exists and is not empty
        if not dist_dir.exists() or not any(dist_dir.iterdir()):
            logger.error("Build completed but dist directory is still missing or empty")
            raise RuntimeError("UI build failed to generate dist directory")
        
        return str(dist_dir)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during UI build: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"UI build failed: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error during UI build: {e}")
        raise