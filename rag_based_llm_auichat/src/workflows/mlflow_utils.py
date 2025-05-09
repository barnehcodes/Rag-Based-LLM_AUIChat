from zenml import step
from zenml.logger import get_logger
import subprocess
import os
import time
import socket
from pathlib import Path

logger = get_logger(__name__)

def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except socket.error:
            return True

@step
def launch_mlflow_dashboard_step() -> str:
    """
    Launches the MLflow UI dashboard as a background process.
    Changes to the notebooks directory and uses the mlruns directory there.
    
    Returns:
        Status message indicating if the dashboard was launched successfully
    """
    logger.info("Attempting to launch MLflow UI dashboard...")
    
    # Path to the notebooks directory
    notebooks_dir = Path("/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/notebooks")
    
    # Ensure the notebooks directory exists
    if not notebooks_dir.exists():
        logger.error(f"Notebooks directory not found at: {notebooks_dir}")
        return "mlflow_launch_failed_directory_not_found"
    
    # Check if mlruns directory exists in notebooks
    mlruns_dir = notebooks_dir / "mlruns"
    if not mlruns_dir.exists():
        logger.warning(f"mlruns directory not found at {mlruns_dir}. It will be created by MLflow.")
    
    # Check if port is already in use
    port = 5001  # Use port 5001 to avoid conflicts with common port 5000
    if is_port_in_use(port):
        logger.info(f"Port {port} is already in use. MLflow UI might already be running.")
        return f"mlflow_port_{port}_already_in_use"
    
    # MLflow command to start the UI server
    backend_store_uri = "file:./mlruns"  # Relative to notebooks_dir
    mlflow_cmd = [
        "mlflow", "ui",
        "--backend-store-uri", backend_store_uri,
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    
    try:
        logger.info(f"Executing MLflow command in {notebooks_dir}: {' '.join(mlflow_cmd)}")
        
        # Start MLflow UI as a background process in the notebooks directory
        process = subprocess.Popen(
            mlflow_cmd,
            cwd=str(notebooks_dir),  # Run in notebooks directory
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is not None:
            # Process terminated already
            stderr = process.stderr.read() if process.stderr else "N/A"
            stdout = process.stdout.read() if process.stdout else "N/A"
            logger.error(f"MLflow UI failed to start. Return code: {process.returncode}")
            logger.error(f"Stderr: {stderr}")
            logger.error(f"Stdout: {stdout}")
            return "mlflow_launch_failed"
        
        # Process is running
        logger.info(f"MLflow UI dashboard started successfully (PID: {process.pid})")
        logger.info(f"Dashboard available at: http://localhost:{port}")
        
        return f"mlflow_dashboard_launched_on_port_{port}"
        
    except FileNotFoundError:
        logger.error("MLflow command not found. Ensure MLflow is installed.")
        return "mlflow_launch_failed_command_not_found"
        
    except Exception as e:
        logger.error(f"Failed to launch MLflow UI: {e}")
        return f"mlflow_launch_failed: {str(e)}"