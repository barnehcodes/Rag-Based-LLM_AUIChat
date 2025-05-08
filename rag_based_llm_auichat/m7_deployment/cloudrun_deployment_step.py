"""
Cloud Run deployment step for AUIChat RAG model with similarity search
This step deploys the AUIChat RAG model to Google Cloud Run with vector similarity search
"""
from zenml import step
from zenml.logger import get_logger
import os
import tempfile
import subprocess
import json
from typing import Dict, Any, Optional

logger = get_logger(__name__)

@step
def deploy_cloudrun_rag_service(
    preprocessed_nodes_path: str,
    collection_name: str = "AUIChatVectoreCol-384",
    project_id: Optional[str] = None,
    region: str = "us-central1",
    service_name: str = "auichat-rag-optimized",
    memory: str = "4Gi",  # Increased memory for embedding model
    cpu: str = "2",       # Increased CPU for better performance
    timeout: str = "300s",
) -> Dict[str, Any]:
    """
    Deploy the optimized RAG service to Cloud Run with vector similarity search
    
    Args:
        preprocessed_nodes_path: Path to the preprocessed nodes file
        collection_name: Qdrant collection name to use for vector search
        project_id: GCP project ID (uses default if None)
        region: GCP region for deployment
        service_name: Cloud Run service name
        memory: Memory allocation for the service
        cpu: CPU allocation for the service
        timeout: Request timeout for the service
        
    Returns:
        Dict with deployment information including service URL
    """
    # Get project ID from environment if not provided
    if not project_id:
        project_id = os.environ.get("PROJECT_ID", "deft-waters-458118-a3")
    
    # Log the deployment parameters
    logger.info(f"Deploying RAG service to Cloud Run:")
    logger.info(f"  Project ID: {project_id}")
    logger.info(f"  Region: {region}")
    logger.info(f"  Service Name: {service_name}")
    logger.info(f"  Using Qdrant collection: {collection_name}")
    
    # Check that the preprocessed nodes file exists
    if not os.path.exists(preprocessed_nodes_path):
        raise FileNotFoundError(f"Preprocessed nodes file not found at {preprocessed_nodes_path}")
    
    # Create a temporary directory for deployment files
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Created temporary directory: {tmp_dir}")
        
        # Copy the preprocessed nodes file to the temp directory
        subprocess.run(
            ["cp", preprocessed_nodes_path, f"{tmp_dir}/preprocessed_nodes.pkl"],
            check=True
        )
        
        # Create the Flask app with vector similarity search using Qdrant
        with open(f"{tmp_dir}/app.py", "w") as f:
            f.write('''
"""
AUIChat RAG Model Service with Vector Embedding Search
This optimized version computes embeddings on-the-fly and uses Qdrant for vector similarity search
"""
import os
import pickle
import logging
import json
import time
import numpy as np
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.http import models
import traceback
from sentence_transformers import SentenceTransformer

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("auichat-service")

# Initialize Flask app
app = Flask(__name__)

# Global variables
nodes = None
qdrant_client = None
embedding_model = None
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "AUIChatVectoreCol-384")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM")
EMBED_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model that produces 384-dim vectors

def load_embedding_model():
    """Load the embedding model for query encoding"""
    global embedding_model
    
    try:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        embedding_model = SentenceTransformer(MODEL_NAME)
        logger.info(f"Successfully loaded embedding model")
        return True
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_nodes():
    """Load the preprocessed nodes as fallback"""
    global nodes
    
    try:
        logger.info("Loading preprocessed nodes from local file...")
        
        # Look in multiple locations for the pickle file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "preprocessed_nodes.pkl"),
            "/app/preprocessed_nodes.pkl"  # Docker container path
        ]
        
        nodes_path = None
        for path in possible_paths:
            if os.path.exists(path):
                nodes_path = path
                break
            
        if not nodes_path:
            logger.error(f"Nodes file not found in any of these locations: {possible_paths}")
            raise FileNotFoundError("Nodes file not found in expected locations")
        
        with open(nodes_path, "rb") as f:
            nodes = pickle.load(f)
            
        logger.info(f"Successfully loaded {len(nodes)} preprocessed nodes from {nodes_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading nodes: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def connect_to_qdrant():
    """Connect to Qdrant vector database cloud instance"""
    global qdrant_client
    
    try:
        logger.info(f"Connecting to Qdrant Cloud at {QDRANT_HOST}...")
        
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
            https=True,
            timeout=30.0  # Increased timeout for reliability
        )
        
        # Test connection by listing collections
        try:
            collections = qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            logger.info(f"Available collections: {collection_names}")
            
            # Check if our target collection exists
            if COLLECTION_NAME in collection_names:
                # Get detailed collection info
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                logger.info(f"Successfully connected to Qdrant collection '{COLLECTION_NAME}'")
                logger.info(f"Collection stats: {collection_info.points_count} vectors, dimension: {collection_info.config.params.vectors.size}")
                return True
            else:
                logger.error(f"Collection '{COLLECTION_NAME}' not found. Available collections: {collection_names}")
                return False
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def vector_search(query_text, top_k=3):
    """Search for relevant vectors in Qdrant using vector embeddings"""
    try:
        logger.info(f"Searching Qdrant collection '{COLLECTION_NAME}' for query: '{query_text}'")
        
        # Generate embedding for the query text
        if embedding_model is None:
            logger.error("Embedding model not loaded. Cannot perform vector search.")
            return []
            
        # Generate query embedding
        start_time = time.time()
        query_embedding = embedding_model.encode(query_text)
        encoding_time = (time.time() - start_time) * 1000
        logger.info(f"Generated query embedding in {encoding_time:.2f}ms")
        
        # Search with the query embedding vector
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=False,  # Don't need the actual vectors
        )
        
        logger.info(f"Qdrant vector search returned {len(search_result)} results")
        
        # Format the results including scores
        results = []
        for i, scored_point in enumerate(search_result):
            point_id = scored_point.id
            score = scored_point.score
            payload = scored_point.payload
            
            # Log each result for debugging
            logger.info(f"Result {i+1}: ID={point_id}, Score={score:.4f}")
            
            # Extract content from payload - try different possible keys with _node_content as priority
            content = None
            
            # First check for _node_content which we saw in the collection testing
            if '_node_content' in payload:
                content = payload['_node_content']
                logger.info(f"Found content in _node_content field")
            else:
                # Check other possible keys
                for key in ['text', 'content', 'page_content', 'chunk']:
                    if key in payload:
                        content = payload[key]
                        logger.info(f"Found content in {key} field")
                        break
            
                if content is None:
                    # If no expected keys found, use the first text field we can find
                    for key, value in payload.items():
                        if isinstance(value, str) and len(value) > 10:
                            content = value
                            logger.info(f"Found content in unexpected key: {key}")
                            break
            
            # If still no content, use a placeholder
            if content is None:
                logger.warning(f"Could not extract content from payload for point {point_id}")
                logger.warning(f"Available keys: {list(payload.keys())}")
                content = "No content available"
            
            # Determine source file
            source_file = None
            if 'file_name' in payload:
                source_file = payload['file_name']
            
            results.append({
                "id": point_id,
                "score": score,
                "text": content,
                "source": source_file
            })
            
        return results
        
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def keyword_search(query, top_k=3):
    """Fallback keyword-based search when Qdrant isn't available"""
    matching_nodes = []
    scores = []
    
    # Split query into keywords and remove common words
    keywords = [w for w in query.lower().split() if len(w) > 3]
    logger.info(f"Searching for keywords: {keywords}")
    
    # Score each node based on keyword matches
    for node in nodes:
        content = node.get_content().lower()
        score = sum(content.count(keyword) for keyword in keywords)
        if score > 0:
            matching_nodes.append(node)
            scores.append(score)
            
    # Sort by score and take top k
    if matching_nodes:
        sorted_pairs = sorted(zip(matching_nodes, scores), key=lambda x: x[1], reverse=True)
        return [(node, float(score)) for node, score in sorted_pairs[:top_k]]
    
    logger.warning(f"No matching nodes found for keywords: {keywords}")
    return []

# Initialize necessary components
logger.info("Starting AUIChat RAG Model Service - Qdrant Connected Version")

# First load the embedding model since we need it for vector search
embedding_model_loaded = load_embedding_model()
if not embedding_model_loaded:
    logger.error("Failed to load embedding model. Vector search will not be available.")

# Connect to Qdrant
qdrant_available = connect_to_qdrant()
nodes_available = False

if not qdrant_available or not embedding_model_loaded:
    logger.warning("Qdrant connection or embedding model failed. Falling back to local nodes for search.")
    nodes_available = load_nodes()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy" if (qdrant_available and embedding_model_loaded) or nodes_available else "degraded",
        "qdrant_available": qdrant_available,
        "embedding_model_loaded": embedding_model_loaded,
        "local_nodes_loaded": nodes_available,
        "using_collection": COLLECTION_NAME
    }
    
    if health_data["status"] == "degraded":
        if not qdrant_available:
            health_data["qdrant_error"] = "Qdrant connection failed"
        if not embedding_model_loaded:
            health_data["embedding_error"] = "Embedding model failed to load"
        if not nodes_available:
            health_data["nodes_error"] = "Local nodes failed to load"
            
    return jsonify(health_data), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint for RAG queries
    Returns semantically relevant nodes that match the query
    """
    if (not qdrant_available or not embedding_model_loaded) and not nodes_available:
        return jsonify({
            "error": "No search capability available. Both Qdrant+embedding and local nodes failed to load.",
            "deployed_model_id": "auichat-rag-optimized" # Corrected ID
        }), 500
    
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
                debug_mode = instance.get("debug", False)
            elif isinstance(instance, str):
                query = instance
                debug_mode = False
            elif isinstance(instance, list) and len(instance) > 0:
                # For numeric arrays (test case), just return sample data
                predictions.append([float(sum(instance))])
                continue
            else:
                try:
                    query = str(instance)
                    debug_mode = False
                except:
                    predictions.append({"error": "Invalid input format"})
                    continue
            
            # Track timing for debug mode
            timings = {}
            search_params = {
                "top_k": 3,
                "method": "unknown"
            }
            
            # Start timing
            start_time = time.time()
            
            # Find relevant content
            matching_texts = []
            
            # Preferred: Qdrant vector search
            if qdrant_available and embedding_model_loaded:
                search_params["method"] = "qdrant_vector_search"
                search_params["collection"] = COLLECTION_NAME
                search_params["embedding_model"] = MODEL_NAME
                
                search_results = vector_search(query, top_k=search_params["top_k"])
                
                # Format results into the expected structure
                matching_texts = [
                    {"text": result["text"], "score": result["score"], "id": result["id"], "source": result.get("source", "Unknown")} 
                    for result in search_results
                ]
                
            # Fallback: Keyword search on local nodes
            elif nodes_available:
                search_params["method"] = "local_keyword_search"
                matching_nodes_with_scores = keyword_search(query, top_k=search_params["top_k"])
                
                # Format results into the expected structure
                matching_texts = [
                    {"text": node.get_content(), "score": float(score), "id": node.node_id if hasattr(node, "node_id") else "unknown", "source": "local_node"} 
                    for node, score in matching_nodes_with_scores
                ]
            
            # Record search time
            search_time = time.time()
            timings["search"] = (search_time - start_time) * 1000  # ms
            
            # Prepare response
            if matching_texts:
                # Format a response with the matching texts
                answer = f"Here's what I found about '{query}':\\n\\n"
                sources = []
                
                for i, result in enumerate(matching_texts):
                    # Truncate text for display in answer
                    display_text = result['text']
                    if len(display_text) > 300:
                        display_text = display_text[:300] + "..."
                    
                    answer += f"Source {i+1}: {display_text}\\n\\n"
                    sources.append(result['text'])
                
                response_data = {
                    "answer": answer,
                    "sources": sources
                }
                
                # Add debug info if requested
                if debug_mode:
                    debug_info = {
                        "search_params": search_params,
                        "timings": timings,
                        "top_chunks": matching_texts,
                        "qdrant_available": qdrant_available,
                        "embedding_model_loaded": embedding_model_loaded,
                        "nodes_available": nodes_available
                    }
                    response_data["debug_info"] = debug_info
                
                predictions.append(response_data)
            else:
                # No results found
                no_results_msg = f"I don't have specific information about '{query}'. Please try another question."
                
                response_data = {
                    "answer": no_results_msg,
                    "sources": []
                }
                
                # Add debug info even for empty results
                if debug_mode:
                    debug_info = {
                        "search_params": search_params,
                        "timings": timings,
                        "error": "No matching content found",
                        "qdrant_available": qdrant_available,
                        "embedding_model_loaded": embedding_model_loaded,
                        "nodes_available": nodes_available
                    }
                    response_data["debug_info"] = debug_info
                
                predictions.append(response_data)
        
        # Return response in Vertex AI-compatible format
        return jsonify({
            "predictions": predictions,
            "deployed_model_id": "auichat-rag-optimized"
        })
        
    except Exception as e:
        logger.error(f"Prediction error details: {str(e)}", exc_info=True) # Log full traceback
        
        # Simplified error response to reduce risk of jsonify failing
        error_response = {
            "error_message": "An unexpected error occurred in the prediction endpoint.",
            "details": str(e), # Simple error message for the client
            "status_check": {
                "qdrant_is_available": qdrant_available,
                "embedding_model_is_loaded": embedding_model_loaded,
                "local_nodes_are_available": nodes_available
            }
        }
        return jsonify(error_response), 500

if __name__ == "__main__":
    # Run the Flask server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
''')
        
        # Create the Dockerfile
        with open(f"{tmp_dir}/Dockerfile", "w") as f:
            f.write('''
# Use Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \\
    build-essential \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PORT=8080
ENV QDRANT_COLLECTION=''' + collection_name + '''
ENV QDRANT_HOST=40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io

# Command to run the application
CMD ["python", "app.py"]
''')
        
        # Create the requirements file with sentence-transformers
        with open(f"{tmp_dir}/requirements.txt", "w") as f:
            f.write('''
flask>=2.0.0
numpy>=1.22.0
scikit-learn>=1.0.0
qdrant-client>=1.4.0
sentence-transformers>=2.2.2
torch>=2.0.0
transformers>=4.30.0
''')
        
        # Build and deploy to Cloud Run
        logger.info("Building container image")
        
        # Set up the image name
        image_name = f"gcr.io/{project_id}/{service_name}"
        
        # Build the container image
        build_cmd = ["gcloud", "builds", "submit", "--tag", image_name, tmp_dir]
        logger.info(f"Running: {' '.join(build_cmd)}")
        
        build_process = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True
        )
        
        if build_process.returncode != 0:
            logger.error(f"Error building container image: {build_process.stderr}")
            raise RuntimeError(f"Failed to build container image: {build_process.stderr}")
            
        logger.info("Container image built successfully")
        
        # Deploy to Cloud Run
        logger.info(f"Deploying to Cloud Run: {service_name}")
        
        deploy_cmd = [
            "gcloud", "run", "deploy", service_name,
            "--image", image_name,
            "--platform", "managed",
            "--region", region,
            "--memory", memory,
            "--cpu", cpu,
            "--timeout", timeout,
            "--allow-unauthenticated"
        ]
        logger.info(f"Running: {' '.join(deploy_cmd)}")
        
        deploy_process = subprocess.run(
            deploy_cmd,
            capture_output=True,
            text=True
        )
        
        if deploy_process.returncode != 0:
            logger.error(f"Error deploying to Cloud Run: {deploy_process.stderr}")
            raise RuntimeError(f"Failed to deploy to Cloud Run: {deploy_process.stderr}")
            
        logger.info("Deployment to Cloud Run successful")
        
        # Get the service URL
        url_cmd = [
            "gcloud", "run", "services", "describe", service_name,
            "--platform", "managed",
            "--region", region,
            "--format", "value(status.url)"
        ]
        
        url_process = subprocess.run(
            url_cmd,
            capture_output=True,
            text=True
        )
        
        if url_process.returncode != 0:
            logger.error(f"Error getting service URL: {url_process.stderr}")
            service_url = "unknown"
        else:
            service_url = url_process.stdout.strip()
            
        # Save deployment info to JSON file
        deployment_info = {
            "service_name": service_name,
            "service_url": service_url,
            "project_id": project_id,
            "region": region,
            "image_name": image_name,
            "collection_name": collection_name
        }
        
        json_path = os.path.expanduser("~/Rag-Based-LLM_AUIChat/cloudrun_optimized_info.json")
        with open(json_path, "w") as f:
            json.dump(deployment_info, f, indent=2)
            
        logger.info(f"Deployment info saved to {json_path}")
        logger.info(f"Service URL: {service_url}")
        
        return deployment_info