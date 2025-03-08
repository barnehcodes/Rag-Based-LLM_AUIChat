from zenml import step
import pickle
from llama_index.core import VectorStoreIndex, StorageContext
from pipeline.config import vector_store, embed_model

@step
def create_and_store_index(nodes_file: str):
    """Loads preprocessed nodes from file, creates an index, and stores embeddings in Qdrant."""
    # Load nodes from the pickle file
    with open(nodes_file, "rb") as f:
        nodes = pickle.load(f)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        store_nodes_override=True
    )
    
    index.storage_context.persist()
    print("✅ All data successfully stored in Qdrant!")
    return "index_created"  # You can return a simple status string
