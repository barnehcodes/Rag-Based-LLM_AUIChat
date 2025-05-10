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
    Launches the MLflow UI dashboard as a background process using nohup to ensure it remains
    running after the pipeline completes.
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
    
    # Create a shell script to launch MLflow UI
    script_content = f"""#!/bin/bash
cd {notebooks_dir}
nohup mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port {port} > mlflow_ui.log 2>&1 &
echo $! > mlflow_ui.pid
echo "MLflow UI started on port {port}. PID saved to mlflow_ui.pid."
echo "Dashboard available at: http://localhost:{port}"
echo "Log file: mlflow_ui.log"
"""
    
    # Write the script to the notebooks directory
    script_path = notebooks_dir / "launch_mlflow_ui.sh"
    try:
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Execute the script
        logger.info(f"Executing MLflow launch script at {script_path}")
        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=str(notebooks_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"MLflow UI dashboard script executed successfully")
            logger.info(result.stdout)
            logger.info(f"Dashboard available at: http://localhost:{port}")
            
            # Write instructions to a file for the user
            instructions = f"""
MLflow Dashboard Instructions:
=============================
The MLflow UI dashboard has been started on port {port}.
You can access it at: http://localhost:{port}

If the dashboard is not accessible, check the log file:
{notebooks_dir}/mlflow_ui.log

To stop the dashboard, run:
kill $(cat {notebooks_dir}/mlflow_ui.pid)
"""
            with open(notebooks_dir / "mlflow_dashboard_instructions.txt", "w") as f:
                f.write(instructions)
                
            return f"mlflow_dashboard_launched_on_port_{port}"
        else:
            logger.error(f"MLflow UI script failed. Return code: {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            return "mlflow_launch_failed"
        
    except Exception as e:
        logger.error(f"Failed to launch MLflow UI: {e}")
        return f"mlflow_launch_failed: {str(e)}"