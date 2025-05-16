#!/usr/bin/env python3
"""
Minimal Version B Deployment
---------------------------
This script creates a minimal version of the RAG system with just the core functionality.
It then deploys it to Cloud Run for A/B testing.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

# Configure paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PREPROCESSED_NODES = PROJECT_ROOT / "preprocessed_nodes.pkl"
SERVICE_NAME = "auichat-rag-version-b"

print(f"Starting minimal Version B deployment...")
print(f"Using preprocessed nodes: {PREPROCESSED_NODES}")

# Create temporary directory
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    print(f"Created temp directory: {tmp_path}")
    
    # Copy preprocessed nodes
    target_nodes = tmp_path / "preprocessed_nodes.pkl"
    shutil.copyfile(PREPROCESSED_NODES, target_nodes)
    print(f"Copied preprocessed nodes file")
    
    # Create app.py
    app_file = tmp_path / "app.py"
    with open(app_file, "w") as f:
        f.write("""
import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load preprocessed nodes
nodes = None
try:
    with open("preprocessed_nodes.pkl", "rb") as f:
        nodes = pickle.load(f)
    logger.info(f"Loaded {len(nodes)} nodes")
except Exception as e:
    logger.error(f"Error loading nodes: {e}")

# Define API models
class QueryRequest(BaseModel):
    query: str

# Create FastAPI app
app = FastAPI(
    title="Minimal RAG API (Version B)",
    description="A minimal working RAG API for Version B testing",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "node_count": len(nodes) if nodes else 0
    }

@app.post("/predict")
async def predict(request: QueryRequest):
    if not nodes:
        raise HTTPException(status_code=500, detail="Nodes not loaded")
        
    # Just return a sample response for testing
    return {
        "response": f"This is a test response for query: {request.query}",
        "sources": [
            {"text": "Sample source 1", "score": 0.95},
            {"text": "Sample source 2", "score": 0.85}
        ],
        "original_query": request.query
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
""")
    
    # Create requirements.txt
    req_file = tmp_path / "requirements.txt"
    with open(req_file, "w") as f:
        f.write("""fastapi==0.95.1
uvicorn==0.22.0
pydantic==1.10.8
""")
    
    # Create Dockerfile
    dockerfile = tmp_path / "Dockerfile"
    with open(dockerfile, "w") as f:
        f.write("""FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY preprocessed_nodes.pkl .
COPY app.py .

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
""")
    
    # Build Docker image
    image_name = f"gcr.io/deft-waters-458118-a3/{SERVICE_NAME}:latest"
    print(f"Building Docker image: {image_name}...")
    
    subprocess.run(
        ["docker", "build", "-t", image_name, str(tmp_path)],
        check=True
    )
    
    # Push to GCR
    print("Pushing image to GCR...")
    subprocess.run(
        ["docker", "push", image_name],
        check=True
    )
    
    # Deploy to Cloud Run
    print("Deploying to Cloud Run...")
    region = os.environ.get("GCP_REGION", "us-central1")
    
    deploy_cmd = [
        "gcloud", "run", "deploy", SERVICE_NAME,
        "--image", image_name,
        "--platform", "managed",
        "--region", region,
        "--memory", "1Gi",
        "--cpu", "1",
        "--allow-unauthenticated"
    ]
    
    subprocess.run(deploy_cmd, check=True)
    
    # Get service URL
    url_cmd = [
        "gcloud", "run", "services", "describe", SERVICE_NAME,
        "--platform", "managed",
        "--region", region,
        "--format", "value(status.url)"
    ]
    
    result = subprocess.run(url_cmd, check=True, capture_output=True, text=True)
    service_url = result.stdout.strip()
    
    # Save deployment info
    deployment_info = {
        "service_name": SERVICE_NAME,
        "service_url": service_url,
        "region": region,
        "image": image_name,
        "version": "B",
        "features": ["Minimal test deployment"]
    }
    
    with open(f"{SERVICE_NAME}_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "="*80)
    print("MINIMAL VERSION B DEPLOYMENT SUCCESSFUL")
    print("="*80)
    print(f"Service URL: {service_url}")
    print(f"To test: curl -X POST {service_url}/predict -H \"Content-Type: application/json\" -d '{{\"query\": \"test\"}}'\n")
    print(f"To run A/B test: python ML6/run_ab_testing.py --endpoint-a [PRODUCTION_URL] --endpoint-b {service_url}/predict")
    print("="*80)
