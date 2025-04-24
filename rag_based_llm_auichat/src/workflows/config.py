# config.py
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex # Added VectorStoreIndex import
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import socket
from dotenv import load_dotenv
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent.parent
# STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage") # Local storage less relevant now
COLLECTION_NAME = "AUIChatVectoreCol-384" # Use the new cloud collection name
EMBED_DIM = 384 # Dimension for BGE model

def load_environment():
    """Load environment variables and initialize settings"""
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Set up llama-index Settings
    from llama_index.core import Settings
    Settings.embed_model = embed_model
    
    # Verify Qdrant connection first
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✅ Connected to Qdrant collection '{COLLECTION_NAME}' with {collection_info.points_count} vectors")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Qdrant collection '{COLLECTION_NAME}': {str(e)}")
        # Optionally check local storage as a fallback if needed, but primary is Qdrant
        # if os.path.exists(STORAGE_DIR) and os.path.isfile(os.path.join(STORAGE_DIR, "index_store.json")):
        #     print(f"⚠️ Qdrant connection failed, but local vector store found at {STORAGE_DIR}")
        #     return True # Or False depending on whether local fallback is desired
        # else:
        #     print(f"❌ Qdrant connection failed and local vector store not found.")
        return False

# Initialize the embedding model with BGE
# BGE is a better model for semantic search and can be used locally
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Try to connect to Qdrant (cloud or local) as a fallback
try:
    # Qdrant connection details - Use cloud instance
    QDRANT_HOST = "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM" # Consider env variable
    
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
        https=True,
        timeout=10.0 # Adjusted timeout
    )
    print(f"✅ Attempting connection to Qdrant cloud at {QDRANT_HOST}")
    qdrant_client.get_collections() # Test connection
    print(f"✅ Successfully connected to Qdrant cloud.")
    
    # Initialize the Qdrant vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client, 
        collection_name=COLLECTION_NAME,
        embed_dim=EMBED_DIM, # Specify embedding dimension
        stores_text=True,
    )
    print(f"✅ Initialized QdrantVectorStore for collection '{COLLECTION_NAME}'")
    
except Exception as e:
    print(f"❌ Error connecting to Qdrant cloud: {str(e)}")
    # Removed fallback to local Qdrant/storage to enforce cloud usage
    print("⚠️ Failed to connect to Qdrant cloud. Vector store operations might fail.")
    qdrant_client = None
    vector_store = None

# Load the index directly from the Qdrant vector store
index = None
if vector_store:
    try:
        print(f"Loading index from Qdrant collection '{COLLECTION_NAME}'...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        print("✅ Successfully loaded index from Qdrant cloud vector store")
    except Exception as e:
        print(f"❌ Error loading index from Qdrant vector store: {str(e)}")
        index = None
else:
    print("❌ Cannot load index because vector store connection failed.")

# Load environment settings (optional, depending on usage)
# load_environment()

