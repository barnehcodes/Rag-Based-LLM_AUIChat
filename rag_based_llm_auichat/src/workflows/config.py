# config.py
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables and initialize settings"""
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Set up llama-index Settings
    from llama_index.core import Settings
    Settings.embed_model = embed_model
    
    # Verify Qdrant connection
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"✅ Connected to Qdrant collection '{COLLECTION_NAME}' with {collection_info.points_count} vectors")
        return True
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {str(e)}")
        return False

# Qdrant connection details - Using cloud instance
QDRANT_HOST = "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "AUIChatVectoreCol"

# Connect to Qdrant cloud
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM",
    https=True
)

# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/msmarco-distilbert-base-v4")

# Initialize the Qdrant vector store (with text storage enabled)
vector_store = QdrantVectorStore(
    client=qdrant_client, 
    collection_name=COLLECTION_NAME,
    text_key="text",
    metadata_key="metadata",
    content_key="content",
    embed_dim=768,  # Must match embedding dimension
    stores_text=True,  # Store full document content in Qdrant
)

