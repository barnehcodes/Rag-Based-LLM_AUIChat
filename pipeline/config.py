# config.py
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
# Qdrant connection details
QDRANT_HOST = "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "AUIChatVectoreCol"

QR_api_key = os.getenv("QDRANT_API_KEY")
# Connect to Qdrant
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    api_key=QR_api_key,
    https=True
)

# (Optional) Check or create collection here if needed.
# For example, you can uncomment and use the code below if desired:
"""
try:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' exists with {collection_info.points_count} vectors")
except Exception:
    print(f"Creating new collection '{COLLECTION_NAME}'")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768,  # msmarco-distilbert-base-v4 embedding size
            distance=models.Distance.COSINE
        )
    )
"""

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
