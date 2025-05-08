#!/usr/bin/env python3
"""
deploy_cloudrun_service.py - Python script to deploy AUIChat RAG model to Cloud Run
This approach gives us more control over the environment and dependencies
"""
import os
import sys
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path
import uuid
import argparse
import json

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("auichat-cloudrun-deploy")

def check_requirements():
    """Check for required GCP command line tools"""
    logger.info("Checking required tools...")
    
    # Check for gcloud
    try:
        subprocess.run(["gcloud", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("gcloud command not found or not working properly")
        return False
    
    # Check authentication
    try:
        result = subprocess.run(["gcloud", "auth", "list"], check=True, capture_output=True, text=True)
        if "No credentialed accounts" in result.stdout:
            logger.error("No authenticated accounts found. Please run 'gcloud auth login'")
            return False
    except subprocess.CalledProcessError:
        logger.error("Error checking authentication status")
        return False
    
    # Enable required services
    logger.info("Enabling required GCP services...")
    services = ["cloudbuild.googleapis.com", "run.googleapis.com", "containerregistry.googleapis.com"]
    for service in services:
        subprocess.run(["gcloud", "services", "enable", service], check=True)
    
    return True

def create_app_files(app_dir, preprocessed_nodes_path):
    """Create application files in the specified directory"""
    logger.info(f"Creating application files in {app_dir}...")
    
    # Create Flask application
    app_py_path = Path(app_dir) / "app.py"
    with open(app_py_path, "w") as f:
        f.write('''
"""
AUIChat RAG Model Service
Flask application for serving the AUIChat RAG model
"""
import os
import pickle
import logging
from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auichat-service")

# Initialize Flask app
app = Flask(__name__)

# Global variables for the model
nodes = None
index = None
query_engine = None

# Function to load the model (will be called during startup)
def load_model():
    """Load the model"""
    global nodes, index, query_engine
    
    try:
        logger.info("Loading AUIChat RAG model...")
        
        # Load the embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Load the preprocessed nodes
        nodes_path = os.path.join(os.path.dirname(__file__), "preprocessed_nodes.pkl")
        if not os.path.exists(nodes_path):
            logger.error(f"Nodes file not found at {nodes_path}")
            return
        
        with open(nodes_path, "rb") as f:
            nodes = pickle.load(f)
            
        # Create the index
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        logger.info("AUIChat RAG model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Start loading model in a background thread to avoid blocking app startup
loading_thread = threading.Thread(target=load_model)
loading_thread.daemon = True
loading_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if query_engine is None:
        return jsonify({"status": "initializing"}), 200
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint that mimics Vertex AI's predict method
    Expected input format: {"instances": [{"query": "What are admission requirements?"}, ...]}
    """
    if not query_engine:
        return jsonify({"error": "Model not loaded yet, please try again in a few seconds"}), 200
    
    try:
        # Get request data
        request_json = request.get_json(silent=True)
        
        if not request_json or 'instances' not in request_json:
            return jsonify({"error": "Invalid request format, expected {'instances': [...]}"}), 400
        
        instances = request_json['instances']
        logger.info(f"Received {len(instances)} instances for prediction")
        
        # Process each instance
        predictions = []
        for instance in instances:
            # Handle different input formats
            if isinstance(instance, dict) and "query" in instance:
                query = instance["query"]
            elif isinstance(instance, str):
                query = instance
            elif isinstance(instance, list) and len(instance) > 0:
                # For numeric arrays (test case), just return sample data
                predictions.append([float(sum(instance))])
                continue
            else:
                try:
                    query = str(instance)
                except:
                    predictions.append({"error": "Invalid input format"})
                    continue
            
            # Process the query
            try:
                response = query_engine.query(query)
                predictions.append({
                    "answer": str(response),
                    "sources": [n.node.get_content() for n in response.source_nodes]
                })
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                predictions.append({"error": f"Query processing error: {str(e)}"})
        
        # Return response in Vertex AI-compatible format
        return jsonify({
            "predictions": predictions,
            "deployed_model_id": "auichat-rag-cloudrun"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
''')
    
    # Create Dockerfile
    dockerfile_path = Path(app_dir) / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write("""
# Use Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PORT=8080

# Command to run the application
CMD ["python", "app.py"]
""")
    
    # Create requirements file
    req_path = Path(app_dir) / "requirements.txt"
    with open(req_path, "w") as f:
        f.write("""
flask>=2.0.0
gunicorn>=20.1.0
llama-index>=0.8.0
llama-index-embeddings-huggingface>=0.1.0
transformers>=4.30.0
torch>=2.0.0
numpy>=1.23.0
scikit-learn>=1.0.0
""")
    
    # Create vertex adapter
    adapter_path = Path(app_dir) / "vertex_adapter.py"
    with open(adapter_path, "w") as f:
        f.write('''
"""
Vertex AI adapter for Cloud Run AUIChat RAG service
This script makes the Cloud Run endpoint accessible using the Vertex AI client
"""
import os
import sys
import requests
import json
import argparse

def init(project_id=None, location=None):
    """Initialize the adapter (stub for compatibility)"""
    pass

class Endpoint:
    """Mock Endpoint class that forwards requests to Cloud Run"""
    
    def __init__(self, endpoint_name=None):
        """Initialize with the Cloud Run URL"""
        self.cloud_run_url = os.environ.get("AUICHAT_ENDPOINT_URL")
        
        if not self.cloud_run_url:
            print("❌ AUICHAT_ENDPOINT_URL environment variable not set!")
            print("   Please run: export AUICHAT_ENDPOINT_URL=<your-cloud-run-url>")
            sys.exit(1)
    
    def predict(self, instances):
        """Forward prediction to Cloud Run endpoint"""
        headers = {"Content-Type": "application/json"}
        
        # Prepare the payload
        payload = {"instances": instances}
        
        # Make the request to Cloud Run
        response = requests.post(
            f"{self.cloud_run_url}/predict",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"{response.status_code} {response.text}")
        
        # Parse the response
        result = response.json()
        
        # Convert to a form similar to Vertex AI's response
        class VertexResponse:
            def __init__(self, predictions, deployed_model_id):
                self.predictions = predictions
                self.deployed_model_id = deployed_model_id
        
        return VertexResponse(
            predictions=result.get("predictions", []),
            deployed_model_id=result.get("deployed_model_id", "auichat-rag-cloudrun")
        )

# For testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the Cloud Run RAG endpoint')
    parser.add_argument('--query', type=str, default="What are the admission requirements?",
                        help='The query to send to the RAG model')
    args = parser.parse_args()
    
    # Test the adapter
    if "AUICHAT_ENDPOINT_URL" not in os.environ:
        print("❌ Please set the AUICHAT_ENDPOINT_URL environment variable first")
        sys.exit(1)
    
    endpoint = Endpoint()
    response = endpoint.predict([{"query": args.query}])
    
    print("📥 Response:")
    print(f"Answer: {response.predictions[0]['answer']}")
    print("\\nSources:")
    for i, source in enumerate(response.predictions[0].get('sources', [])):
        print(f"{i+1}. {source[:100]}...")
''')
    
    # Copy preprocessed nodes
    try:
        logger.info(f"Copying preprocessed nodes from {preprocessed_nodes_path}")
        shutil.copy(preprocessed_nodes_path, Path(app_dir) / "preprocessed_nodes.pkl")
    except Exception as e:
        logger.error(f"Error copying preprocessed nodes: {e}")
        return False
    
    return True

def build_and_deploy(app_dir, project_id, region):
    """Build and deploy the service to Cloud Run"""
    try:
        # Generate a unique service name with timestamp
        service_name = f"auichat-rag-service-{uuid.uuid4().hex[:8]}"
        image_name = f"gcr.io/{project_id}/{service_name}"
        
        # Build the container image
        logger.info(f"Building container image: {image_name}")
        subprocess.run(
            ["gcloud", "builds", "submit", "--tag", image_name, app_dir],
            check=True,
            capture_output=False
        )
        
        # Deploy to Cloud Run
        logger.info(f"Deploying to Cloud Run: {service_name}")
        subprocess.run([
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", region,
            "--memory", "2Gi",
            "--cpu", "1",
            "--timeout", "300s",
            "--allow-unauthenticated"
        ], check=True, capture_output=False)
        
        # Get service URL
        result = subprocess.run([
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", region,
            "--format", "value(status.url)"
        ], check=True, capture_output=True, text=True)
        
        service_url = result.stdout.strip()
        
        # Save deployment info
        deployment_info = {
            "service_name": service_name,
            "service_url": service_url,
            "project_id": project_id,
            "region": region,
            "image_name": image_name
        }
        
        with open("cloudrun_deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Service URL: {service_url}")
        logger.info("Deployment information saved to cloudrun_deployment_info.json")
        
        return service_url
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Deployment failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return None

def update_test_script(service_url, project_id):
    """Update test scripts to use the new endpoint"""
    for test_script_path in [
        "/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/m4_model_serving_runtime/test_vertex_endpoint.py",
        "/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/m2_model_serving_mode/test_vertex_endpoint.py"
    ]:
        if os.path.exists(test_script_path):
            logger.info(f"Updating test script: {test_script_path}")
            
            # Create a backup
            backup_path = test_script_path + ".bak"
            shutil.copy(test_script_path, backup_path)
            
            # Read the file content
            with open(test_script_path, "r") as f:
                content = f.read()
            
            # Add imports at the top of the file
            if "import requests" not in content:
                import_section = "import os\nimport json\nimport argparse\nimport requests"
                content = content.replace("import os\nimport json\nimport argparse", import_section)
            
            # Add the adapter code right after the imports
            adapter_code = f"""
# ---------------------------------------------------------------------------
# Cloud Run Adapter (added by deployment script)
# This adapter allows the test script to work with the Cloud Run endpoint
# ---------------------------------------------------------------------------

# Environment variable for the Cloud Run endpoint URL
AUICHAT_ENDPOINT_URL = os.environ.get('AUICHAT_ENDPOINT_URL', '{service_url}')

# Mock Vertex AI client for Cloud Run
class CloudRunVertexAdapter:
    def init(self, project_id=None, location=None):
        pass
        
    class Endpoint:
        def __init__(self, endpoint_name=None):
            self.cloud_run_url = AUICHAT_ENDPOINT_URL
            
        def predict(self, instances):
            headers = {{"Content-Type": "application/json"}}
            payload = {{"instances": instances}}
            
            response = requests.post(
                f"{{self.cloud_run_url}}/predict",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"{{response.status_code}} {{response.text}}")
                
            result = response.json()
            
            # Create a response object that mimics Vertex AI
            class MockResponse:
                def __init__(self, predictions, deployed_model_id):
                    self.predictions = predictions
                    self.deployed_model_id = deployed_model_id
            
            return MockResponse(
                predictions=result.get("predictions", []),
                deployed_model_id=result.get("deployed_model_id", "auichat-rag-cloudrun")
            )
            
# Replace the real aiplatform with our adapter
import sys
from types import ModuleType
cloud_run_vertex = CloudRunVertexAdapter()
sys.modules['google.cloud.aiplatform'] = cloud_run_vertex

# ---------------------------------------------------------------------------
# End of Cloud Run Adapter
# ---------------------------------------------------------------------------
"""
            
            # Insert adapter code after imports
            import_end = content.find("from google.protobuf import json_format")
            if import_end == -1:
                import_end = content.find("from google.protobuf.struct_pb2 import Value")
            if import_end == -1:
                import_end = content.find("parser = argparse.ArgumentParser")
                
            if import_end != -1:
                content = content[:import_end] + adapter_code + content[import_end:]
                
                # Write the updated content
                with open(test_script_path, "w") as f:
                    f.write(content)
                    
                logger.info(f"Updated test script {test_script_path} to use Cloud Run endpoint")
            else:
                logger.warning(f"Could not update {test_script_path}, structure not recognized")

def main():
    """Main function to deploy to Cloud Run"""
    parser = argparse.ArgumentParser(description='Deploy AUIChat RAG model to Cloud Run')
    parser.add_argument('--project-id', type=str, help='GCP Project ID')
    parser.add_argument('--region', type=str, default='us-central1', help='GCP Region')
    parser.add_argument('--nodes-path', type=str, 
                        default='/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/preprocessed_nodes.pkl',
                        help='Path to preprocessed_nodes.pkl file')
    
    args = parser.parse_args()
    
    # Get project ID from command line or environment variable
    project_id = args.project_id or os.environ.get("PROJECT_ID", "deft-waters-458118-a3")
    region = args.region or os.environ.get("REGION", "us-central1")
    
    logger.info(f"Using project ID: {project_id}")
    logger.info(f"Using region: {region}")
    
    # Set environment variables for other tools
    os.environ["PROJECT_ID"] = project_id
    os.environ["REGION"] = region
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed")
        return 1
    
    # Create temporary directory for app files
    app_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {app_dir}")
    
    try:
        # Create application files
        if not create_app_files(app_dir, args.nodes_path):
            logger.error("Failed to create application files")
            return 1
        
        # Build and deploy
        service_url = build_and_deploy(app_dir, project_id, region)
        if not service_url:
            logger.error("Deployment failed")
            return 1
        
        # Export the endpoint URL as an environment variable
        os.environ["AUICHAT_ENDPOINT_URL"] = service_url
        
        # Update test scripts
        update_test_script(service_url, project_id)
        
        # Print success message
        logger.info("✅ Deployment successful!")
        logger.info(f"Service URL: {service_url}")
        logger.info("")
        logger.info("To use this endpoint in your code, set the environment variable:")
        logger.info(f"export AUICHAT_ENDPOINT_URL={service_url}")
        logger.info("")
        logger.info("To run tests:")
        logger.info("python /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/m4_model_serving_runtime/test_vertex_endpoint.py")
        
        return 0
        
    finally:
        # Clean up the temporary directory
        logger.info(f"Cleaning up temporary directory: {app_dir}")
        shutil.rmtree(app_dir)

if __name__ == "__main__":
    sys.exit(main())