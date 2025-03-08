# indexing.py
from llama_index.core import VectorStoreIndex, StorageContext
from config import vector_store, embed_model

def create_and_store_index(nodes):
    # Create storage context using the Qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index with nodes and force storing node content in Qdrant
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        store_nodes_override=True  # Ensure node content is stored in the vector DB
    )
    
    print("All data successfully stored in Qdrant!")
    return index
