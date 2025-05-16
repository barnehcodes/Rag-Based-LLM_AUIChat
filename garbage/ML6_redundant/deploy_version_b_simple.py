#!/usr/bin/env python3
"""
Simplified Version B Deployment
------------------------------
A simplified version of the deploy_version_b.py script that focuses on getting
the core functionality working without complex error handling.
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Default constants
SERVICE_NAME = "auichat-rag-version-b"
PREPROCESSED_NODES = str(PROJECT_ROOT / "preprocessed_nodes.pkl")

def main():
    """Main entry point for the simplified deployment script"""
    print(f"Starting deployment of Version B RAG system to Cloud Run")
    
    # Create temporary directory for deployment files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Created temporary directory: {tmp_path}")

        # Create Dockerfile
        dockerfile_path = tmp_path / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write("""FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy preprocessed nodes and application code
COPY preprocessed_nodes.pkl .
COPY app.py .

# Run the application
ENV QDRANT_COLLECTION=AUIChatVectorCol-384
ENV PREPROCESSED_NODES=/app/preprocessed_nodes.pkl
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
""")
        
        # Create requirements.txt with explicit CPU torch version
        requirements_path = tmp_path / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write("""fastapi==0.95.1
uvicorn==0.22.0
qdrant-client==1.4.0
llama-index==0.8.54
sentence-transformers==2.2.2
torch==2.0.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
transformers==4.30.2
numpy==1.24.3
pydantic==1.10.8
""")
        
        # Copy preprocessed nodes
        import shutil
        preprocessed_nodes_path = tmp_path / "preprocessed_nodes.pkl"
        shutil.copyfile(PREPROCESSED_NODES, preprocessed_nodes_path)
        
        # Create simplified app with basic RAG and Version B enhancements
        app_path = tmp_path / "app.py"
        with open(app_path, "w") as f:
            f.write("""import os
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

try:
    from llama_index.retrievers import BM25Retriever, VectorIndexRetriever, HybridRetriever
except ImportError:
    from llama_index.core.retrievers import BM25Retriever, VectorIndexRetriever, HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI request model
class QueryRequest(BaseModel):
    query: str

# Create FastAPI app
app = FastAPI(
    title="AUIChat Enhanced RAG API (Version B)",
    description="Improved RAG with hybrid retrieval and re-ranking",
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

# Global variables
index = None
llm = None
query_engine = None
cross_encoder = None

@app.on_event("startup")
async def startup():
    global index, llm, query_engine, cross_encoder
    
    try:
        logger.info("Loading models and initializing RAG components...")
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        
        # Initialize LLM with lower temperature for better factuality
        llm = HuggingFaceLLM(
            model_name="google/gemma-2b",
            tokenizer_name="google/gemma-2b", 
            context_window=2048,
            max_new_tokens=512,
            model_kwargs={"temperature": 0.3},
            generate_kwargs={"do_sample": True}
        )
        Settings.llm = llm
        
        # Initialize cross-encoder for re-ranking
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Try to connect to Qdrant or use local nodes as fallback
        try:
            from qdrant_client import QdrantClient
            collection_name = os.environ.get("QDRANT_COLLECTION", "AUIChatVectorCol-384")
            qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant-service:6333")
            
            logger.info(f"Connecting to Qdrant at {qdrant_url}")
            client = QdrantClient(url=qdrant_url)
            
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                raise ValueError(f"Collection {collection_name} not found")
                
            vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
            logger.info(f"Successfully connected to Qdrant collection: {collection_name}")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant: {e}")
            logger.info("Falling back to local nodes")
            
            # Load from the preprocessed nodes pickle file
            nodes_file = os.environ.get("PREPROCESSED_NODES", "/app/preprocessed_nodes.pkl")
            
            with open(nodes_file, 'rb') as f:
                nodes = pickle.load(f)
            
            logger.info(f"Loaded {len(nodes)} nodes from {nodes_file}")
            index = VectorStoreIndex(nodes)
            vector_store = index.vector_store
        
        # Create storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from vector store if needed
        if index is None:
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        
        # Create vector retriever
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=7
        )
        
        # Create BM25 retriever
        all_nodes = list(index.docstore.docs.values())
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=7
        )
        
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=10,
            weights=[0.7, 0.3]
        )
        
        # Create query engine
        query_engine = index.as_query_engine(
            retriever=hybrid_retriever,
            llm=llm,
            similarity_top_k=5
        )
        
        logger.info("RAG system initialization complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

def rerank_nodes(nodes, query):
    """Use cross-encoder to rerank nodes for better precision"""
    if not nodes or not cross_encoder:
        return nodes
    
    # Create pairs of query and document text
    pairs = [(query, node.text) for node in nodes]
    
    # Get scores from cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Assign new scores to nodes
    for node, score in zip(nodes, scores):
        node.score = score
    
    # Sort by new score and take top 5
    reranked = sorted(nodes, key=lambda x: x.score, reverse=True)[:5]
    return reranked

@app.post("/predict")
async def predict(request: QueryRequest):
    """Handle prediction request"""
    if not query_engine:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    query = request.query.strip()
    if not query:
        return {"error": "Empty query", "response": "Please provide a question."}
    
    try:
        # Get response from query engine
        response = query_engine.query(query)
        
        # Get and rerank the source nodes
        source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        reranked_nodes = rerank_nodes(source_nodes, query)
        
        # Format sources
        sources = []
        for node in reranked_nodes:
            if hasattr(node, "metadata") and node.metadata:
                source = {
                    "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    "score": float(node.score) if hasattr(node, "score") else 0.0
                }
                
                if "file_name" in node.metadata:
                    source["file_name"] = node.metadata["file_name"]
                elif "source" in node.metadata:
                    source["file_name"] = node.metadata["source"]
                
                sources.append(source)
        
        return {
            "response": response.response,
            "sources": sources,
            "original_query": query
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "model": "Enhanced RAG (Version B)"
    }
""")
        
        # Build Docker image
        print("Building Docker image...")
        service_name = os.environ.get("SERVICE_NAME", SERVICE_NAME)
        image_name = f"gcr.io/deft-waters-458118-a3/{service_name}:latest"
        
        subprocess.run(
            ["docker", "build", "-t", image_name, str(tmp_path)],
            check=True
        )
        
        # Push to GCR
        print("Pushing image to Google Container Registry...")
        subprocess.run(
            ["docker", "push", image_name],
            check=True
        )
        
        # Deploy to Cloud Run
        print("Deploying to Cloud Run...")
        region = os.environ.get("GCP_REGION", "us-central1")
        
        deploy_cmd = [
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", region,
            "--memory", "2Gi",
            "--cpu", "2",
            "--allow-unauthenticated"
        ]
        
        subprocess.run(deploy_cmd, check=True)
        
        # Get service URL
        url_cmd = [
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", region,
            "--format", "value(status.url)"
        ]
        
        result = subprocess.run(url_cmd, check=True, capture_output=True, text=True)
        service_url = result.stdout.strip()
        
        # Save deployment info
        deployment_info = {
            "service_name": service_name,
            "service_url": service_url,
            "region": region,
            "image": image_name,
            "version": "B",
            "features": [
                "Hybrid retrieval (vector + BM25)",
                "Cross-encoder reranking",
                "Lower temperature (0.3) for better factuality"
            ]
        }
        
        with open(f"{service_name}_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        print("\n" + "="*80)
        print("DEPLOYMENT SUCCESSFUL")
        print("="*80)
        print(f"Service URL: {service_url}")
        print(f"To test: curl -X POST {service_url}/predict -H \"Content-Type: application/json\" -d '{{\"query\": \"What are AUI admission requirements?\"}}'")
        print(f"To run A/B test: python ML6/run_ab_testing.py --endpoint-a [PRODUCTION_URL] --endpoint-b {service_url}/predict")
        print("="*80)

if __name__ == "__main__":
    main()
