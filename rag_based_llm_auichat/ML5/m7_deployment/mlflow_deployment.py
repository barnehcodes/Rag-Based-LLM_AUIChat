"""
MLflow model deployment step for AUIChat
Provides a local MLflow deployment step that doesn't require Seldon or cloud providers
"""
from zenml import step
from zenml.logger import get_logger
import mlflow
import os
import subprocess
import time
import signal
import threading
import atexit

logger = get_logger(__name__)

# Dictionary to keep track of running MLflow server processes
deployed_servers = {}

def _start_mlflow_server(model_uri, port=5500):
    """
    Start a local MLflow model server for the specified model URI
    
    Args:
        model_uri: MLflow model URI to serve
        port: HTTP port to listen on
        
    Returns:
        Tuple of (process, url) where process is the server process and url is the endpoint
    """
    cmd = [
        "mlflow", "models", "serve",
        "--model-uri", model_uri,
        "--port", str(port),
        "--no-conda",
        "--enable-mlserver"
    ]
    
    logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the server time to start
    time.sleep(5)
    
    # Check if the process is still running
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"MLflow server failed to start: {stderr}")
        raise RuntimeError(f"MLflow server failed to start: {stderr}")
    
    url = f"http://localhost:{port}/invocations"
    logger.info(f"MLflow server started at {url}")
    
    return (process, url)

def _cleanup_servers():
    """Clean up all running MLflow servers on exit"""
    for model_name, (process, _) in deployed_servers.items():
        logger.info(f"Stopping MLflow server for {model_name}")
        if process and process.poll() is None:  # If process exists and is running
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

# Register the cleanup function to be called when the program exits
atexit.register(_cleanup_servers)

@step
def deploy_mlflow_model(model_uri: str, port: int = 5500) -> str:
    """
    Deploys the saved MLflow model as a local REST API endpoint
    
    Args:
        model_uri: The MLflow model URI (runs:/<run_id>/<model_name>)
        port: The port to deploy the model on
        
    Returns:
        The URL of the deployed model endpoint
    """
    model_name = model_uri.split('/')[-1]
    
    # Check if a server for this model is already running
    if model_name in deployed_servers:
        old_process, old_url = deployed_servers[model_name]
        if old_process.poll() is None:  # If the process is still running
            logger.info(f"Model {model_name} is already deployed at {old_url}")
            return old_url
        else:
            logger.info(f"Previous deployment for {model_name} has died, redeploying")
    
    # Start a new server
    process, url = _start_mlflow_server(model_uri, port)
    
    # Store the process and URL
    deployed_servers[model_name] = (process, url)
    
    # Start a monitoring thread to ensure the server keeps running
    def monitor_server():
        while True:
            time.sleep(30)  # Check every 30 seconds
            if process.poll() is not None:  # If the process has exited
                logger.error(f"MLflow server for {model_name} has died unexpectedly")
                break
    
    monitor_thread = threading.Thread(target=monitor_server, daemon=True)
    monitor_thread.start()
    
    return url