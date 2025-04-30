#!/usr/bin/env python3
"""
Script to create a vector store on a Qdrant cloud instance with embeddings from PDF files
Uses BGE-small-en-v1.5 model for embeddings
"""
import os
import sys
import glob
import time
import traceback
from typing import List, Dict, Any
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
    # Start with simpler imports
    from llama_index.readers.file import PDFReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.storage.storage_context import StorageContext
    
    # Import Qdrant modules
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
    logger.info("Qdrant modules successfully imported")

    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.info("Ensure llama-index, llama-index-readers-file, transformers, qdrant-client are installed.")
    sys.exit(1) # Exit if essential modules are missing

# Settings
PDF_DIR = os.path.join(script_dir, "raw")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
COLLECTION_NAME = "AUIChatVectoreCol-384" # Updated collection name
EMBED_DIM = 384  # BGE-small dimension
# OUTPUT_DIR = os.path.join(script_dir, "storage") # No longer needed for local storage

# Qdrant Cloud Connection Details (Update with your actual credentials if different)
# It's highly recommended to load sensitive keys from environment variables
# QDRANT_HOST = os.getenv("QDRANT_HOST", "YOUR_QDRANT_HOST")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY")
# Using hardcoded values for now, based on config.py
QDRANT_HOST = "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM"

# Embedding Model
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def process_documents(pdf_files):
    """Process PDF files into document chunks"""
    all_nodes = []
    pdf_reader = PDFReader()
    
    # Process each PDF file
    for pdf_file in pdf_files:
        file_name = os.path.basename(pdf_file)
        logger.info(f"Processing: {file_name}")
        
        try:
            # Load document
            logger.info(f"  Reading document...")
            docs = pdf_reader.load_data(pdf_file)
            
            # Split into chunks
            logger.info(f"  Splitting into chunks...")
            splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            nodes = splitter.get_nodes_from_documents(docs)
            
            # Add file metadata
            for node in nodes:
                if hasattr(node, 'metadata'):
                    node.metadata["file_name"] = file_name
                    node.metadata["source"] = pdf_file
            
            logger.info(f"  Created {len(nodes)} chunks from {file_name}")
            all_nodes.extend(nodes)
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_name}: {str(e)}")
            traceback.print_exc()
    
    return all_nodes

def get_embedding_model(model_name: str):
    """Initialize the HuggingFace embedding model"""
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        # Ensure device is auto-detected or set appropriately if needed
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        logger.info("✅ Embedding model initialized successfully.")
        return embed_model
    except Exception as e:
        logger.error(f"❌ Failed to initialize embedding model: {str(e)}")
        traceback.print_exc()
        return None

def get_qdrant_cloud_client():
    """Connect to the specified Qdrant cloud instance"""
    if not QDRANT_HOST or not QDRANT_API_KEY:
        logger.error("❌ Qdrant host or API key not configured.")
        return None
    try:
        logger.info(f"Connecting to Qdrant cloud at {QDRANT_HOST}...")
        client = QdrantClient(
            host=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
            https=True, # Assuming HTTPS is needed for cloud
            timeout=20.0 # Increased timeout for cloud operations
        )
        client.get_collections() # Test connection
        logger.info(f"✅ Connected to Qdrant cloud instance at {QDRANT_HOST}")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant cloud instance: {str(e)}")
        traceback.print_exc()
        return None

def build_vector_store(nodes, embed_model):
    """Build vector store from document nodes using the Qdrant cloud instance"""
    qdrant_client = get_qdrant_cloud_client()
    
    if not qdrant_client:
        logger.error("❌ Cannot proceed without a Qdrant connection.")
        return None

    logger.info(f"Using Qdrant cloud collection: {COLLECTION_NAME}")
    try:
        # Check if collection exists
        collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)
        
        if collection_exists:
            logger.warning(f"Collection '{COLLECTION_NAME}' already exists. Recreating it...")
            qdrant_client.delete_collection(COLLECTION_NAME)
            time.sleep(1) # Give Qdrant a moment
            
        # Create new collection
        logger.info(f"Creating new collection '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
        
        # Create Qdrant vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embed_dim=EMBED_DIM, # Ensure embed_dim is passed if needed by the version
            stores_text=True, # Store text within Qdrant payload
        )
        
        # Create storage context with Qdrant
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from nodes - this will embed and upload
        logger.info(f"Building index and uploading {len(nodes)} nodes to Qdrant...")
        index = VectorStoreIndex(
            nodes=nodes, 
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True # Show progress during embedding/upload
        )
        
        logger.info(f"✅ Successfully built index and uploaded data to Qdrant collection '{COLLECTION_NAME}'.")
        return index
        
    except Exception as e:
        logger.error(f"❌ Error during Qdrant index creation/upload: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function to run the embedding and upload process"""
    logger.info(f"Starting vector store builder...")
    logger.info(f"PDF directory: {PDF_DIR}")
    
    start_time = time.time()
    
    # Test that PDF directory exists
    if not os.path.exists(PDF_DIR):
        logger.error(f"❌ PDF directory does not exist: {PDF_DIR}")
        return
    
    # Load PDF files
    logger.info(f"Loading PDF files from {PDF_DIR}...")
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    if not pdf_files:
        logger.error(f"❌ No PDF files found in {PDF_DIR}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Initialize embedding model
    logger.info("Initializing BGE embedding model...")
    embed_model = get_embedding_model(EMBED_MODEL_NAME)
    
    # Process documents to get nodes
    nodes = process_documents(pdf_files)
    logger.info(f"Total chunks created: {len(nodes)}")
    
    # Build vector store from nodes
    index = build_vector_store(nodes, embed_model)
    
    if index:
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Completed in {elapsed_time:.2f} seconds")
        logger.info(f"Vector store created/updated in Qdrant cloud collection: {COLLECTION_NAME}")
    else:
        logger.error("❌ Vector store creation failed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"❌ Unhandled exception: {str(e)}")
        traceback.print_exc()