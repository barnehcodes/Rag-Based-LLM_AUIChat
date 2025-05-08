#!/usr/bin/env python3
"""
Script to demonstrate querying the "auichatcloudcol" collection in Qdrant
"""
import os
import sys
import json
import time
from pathlib import Path

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project directory to path
script_dir = Path(os.path.abspath(__file__)).parent
sys.path.append(str(script_dir))

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    sys.exit(1)

# Settings
COLLECTION_NAME = "auichatcloudcol"  # Our new collection
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # BGE-small dimension

# Qdrant Cloud Connection Details
QDRANT_HOST = "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM"

def get_embedding_model():
    """Initialize the HuggingFace embedding model"""
    logger.info(f"Initializing embedding model: {EMBED_MODEL_NAME}")
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        logger.info("✅ Embedding model initialized successfully.")
        return embed_model
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedding model: {str(e)}")
        return None

def get_qdrant_client():
    """Connect to the Qdrant cloud instance"""
    try:
        logger.info(f"Connecting to Qdrant cloud at {QDRANT_HOST}...")
        client = QdrantClient(
            host=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
            https=True,
            timeout=20.0
        )
        
        # Check if our collection exists
        if not client.collection_exists(COLLECTION_NAME):
            logger.error(f"❌ Collection '{COLLECTION_NAME}' does not exist.")
            return None
            
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ Connected to '{COLLECTION_NAME}' collection with {collection_info.points_count} points")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant cloud: {str(e)}")
        return None

def query_qdrant_directly(query_text, limit=5):
    """Query Qdrant directly using the embedding model"""
    client = get_qdrant_client()
    embed_model = get_embedding_model()
    
    if not client or not embed_model:
        return None
        
    # Generate embedding for the query
    query_vector = embed_model.get_text_embedding(query_text)
    
    # Search Qdrant collection
    logger.info(f"Querying Qdrant with: '{query_text}'")
    start_time = time.time()
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Query completed in {elapsed_time:.4f} seconds")
    
    return results

def query_using_llama_index(query_text, top_k=3):
    """Query using LlamaIndex's query engine"""
    client = get_qdrant_client()
    embed_model = get_embedding_model()
    
    if not client or not embed_model:
        return None
        
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embed_dim=EMBED_DIM,
    )
    
    # Create index
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    
    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
    )
    
    # Execute query
    logger.info(f"Querying LlamaIndex with: '{query_text}'")
    start_time = time.time()
    
    response = query_engine.query(query_text)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Query completed in {elapsed_time:.4f} seconds")
    
    return response

def display_payload_content(payload):
    """Extract and display text content from various payload formats"""
    text = None
    file_name = "Unknown"
    
    if 'text' in payload:
        text = payload.get('text')
        file_name = payload.get('metadata', {}).get('file_name', payload.get('file_name', 'Unknown'))
    elif 'metadata' in payload and 'text' in payload['metadata']:
        text = payload['metadata']['text']
        file_name = payload['metadata'].get('file_name', 'Unknown')
    elif '_node_content' in payload:
        try:
            node_content = json.loads(payload.get('_node_content', '{}'))
            text = node_content.get("text", "No text found")
            file_name = node_content.get('metadata', {}).get('file_name', 'Unknown')
        except json.JSONDecodeError:
            text = "Error decoding _node_content"
    elif 'content' in payload:
        content = payload['content']
        if isinstance(content, str):
            try:
                content_data = json.loads(content)
                text = content_data.get('text', content)
            except json.JSONDecodeError:
                text = content
        else:
            text = str(content)
        file_name = payload.get('file_name', 'Unknown')
    
    if text:
        return file_name, text
    else:
        return file_name, f"No text found. Available keys: {list(payload.keys())}"

def main():
    """Main function to demonstrate searching"""
    print("\n" + "="*60)
    print("QDRANT COLLECTION QUERY DEMO")
    print("="*60)
    
    # Sample queries to test
    queries = [
        "What are the admission requirements for freshmen?",
        "Tell me about the PiP program at AUI.",
        "What services does the counseling center offer?",
        "How can I transfer to AUI from another university?"
    ]
    
    # Method 1: Direct Qdrant queries
    print("\n[Method 1] Direct Qdrant queries with BGE embeddings\n")
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'\n{'-'*50}")
        
        results = query_qdrant_directly(query, limit=3)
        
        if not results:
            print("No results found or query failed.")
            continue
            
        # Display results
        for j, result in enumerate(results):
            print(f"\nResult {j+1} - Score: {result.score:.4f}")
            file_name, text = display_payload_content(result.payload)
            print(f"From: {file_name}")
            print(f"Text: {text[:300]}...")
    
    # Method 2: LlamaIndex query engine
    print("\n\n[Method 2] LlamaIndex query engine\n")
    
    for i, query in enumerate(queries):
        print(f"\nQuery {i+1}: '{query}'\n{'-'*50}")
        
        response = query_using_llama_index(query)
        
        if not response:
            print("No results found or query failed.")
            continue
            
        # Display response
        print(f"Response: {response}\n")
        print("Sources:")
        
        for j, node in enumerate(response.source_nodes):
            print(f"\nSource {j+1} - Score: {node.score:.4f}")
            print(f"From: {node.metadata.get('file_name', 'Unknown')}")
            print(f"Text: {node.text[:300]}...")

if __name__ == "__main__":
    main()