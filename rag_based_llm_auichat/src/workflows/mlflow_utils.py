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

def wait_for_port_to_be_active(port: int, timeout: int = 10) -> bool:
    """Wait for a port to become active (accepting connections).
    
    Args:
        port: Port number to check
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if port is active, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:  # Port is open and accepting connections
                    return True
        except socket.error:
            pass
        time.sleep(0.5)
    return False

def install_mlflow_dependencies():
    """Ensure MLflow UI dependencies are installed."""
    try:
        logger.info("Installing/verifying MLflow UI dependencies...")
        # Install common dependencies that might be missing
        subprocess.run(
            ["pip", "install", "jinja2>=2.11.3", "flask>=2.0.0", "werkzeug>=2.0.0", "markupsafe>=2.0.0"],
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install MLflow UI dependencies: {e}")
        return False

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
    
    # Install dependencies if needed
    install_mlflow_dependencies()
    
    # Path to the notebooks directory
    notebooks_dir = Path("/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/notebooks")
    
    # Ensure the notebooks directory exists
    if not notebooks_dir.exists():
        logger.error(f"Notebooks directory not found at: {notebooks_dir}")
        return "mlflow_launch_failed_directory_not_found"
    
    # Ensure mlruns directory exists
    mlruns_dir = notebooks_dir / "mlruns"
    if not mlruns_dir.exists():
        logger.info(f"Creating mlruns directory at {mlruns_dir}")
        mlruns_dir.mkdir(exist_ok=True)
    
    # Check if port is already in use
    port = 5001  # Use port 5001 to avoid conflicts with common port 5000
    if is_port_in_use(port):
        logger.info(f"Port {port} is already in use. MLflow UI might already be running.")
        # Test if MLflow is actually responding on this port
        if wait_for_port_to_be_active(port, 2):
            logger.info(f"Confirmed MLflow UI is already running on port {port}")
            return f"mlflow_port_{port}_already_active"
        else:
            # Port is in use but not by MLflow - try killing whatever is using it
            try:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
                logger.info(f"Killed process using port {port}")
                time.sleep(1)  # Give it time to release the port
            except Exception as e:
                logger.warning(f"Failed to kill process on port {port}: {e}")
                # Try another port
                port = 5002
                logger.info(f"Trying alternative port {port}")
                if is_port_in_use(port):
                    logger.error(f"Alternative port {port} is also in use. Cannot launch MLflow UI.")
                    return "mlflow_launch_failed_all_ports_in_use"
    
    # Launch MLflow UI directly
    logger.info(f"Launching MLflow UI on port {port} in {notebooks_dir}")
    try:
        # Start MLflow in a separate process that will continue running after this script ends
        cmd = [
            "mlflow", "ui", 
            "--backend-store-uri", f"file:{mlruns_dir}", 
            "--host", "0.0.0.0", 
            "--port", str(port)
        ]
        
        # Use subprocess.Popen with nohup to ensure the process stays alive
        process = subprocess.Popen(
            f"nohup {' '.join(cmd)} > {notebooks_dir}/mlflow_ui.log 2>&1 &",
            shell=True,
            cwd=str(notebooks_dir),
        )
        
        # Wait a moment for the process to start
        time.sleep(2)
        
        # Check if the process is running and port is active
        if wait_for_port_to_be_active(port, 10):
            logger.info(f"✓ MLflow UI started successfully on port {port}")
            
            # Get the PID of the mlflow process for later use
            try:
                # Find PID of the process using the port
                result = subprocess.run(
                    ["lsof", "-i", f":{port}", "-t"],
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    pid = result.stdout.strip()
                    with open(notebooks_dir / "mlflow_ui.pid", "w") as f:
                        f.write(pid)
                    logger.info(f"MLflow UI process PID: {pid}, saved to mlflow_ui.pid")
            except Exception as e:
                logger.warning(f"Could not determine MLflow UI PID: {e}")
            
            # Write instructions to a file for the user
            instructions = f"""
MLflow Dashboard Instructions:
=============================
The MLflow UI dashboard is running on port {port}.
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
            # Check the log file for errors
            if (notebooks_dir / "mlflow_ui.log").exists():
                with open(notebooks_dir / "mlflow_ui.log", "r") as f:
                    log_content = f.read()
                logger.error(f"MLflow UI failed to start. Log file contents: {log_content[:500]}")
            
            return "mlflow_launch_failed_process_not_responding"
            
    except Exception as e:
        logger.error(f"Failed to launch MLflow UI: {e}")
        return f"mlflow_launch_failed: {str(e)}"