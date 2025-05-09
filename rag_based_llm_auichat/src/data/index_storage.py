from zenml import step
import pickle
from llama_index.core import VectorStoreIndex, StorageContext
from src.workflows.config.config import vector_store, embed_model, COLLECTION_NAME, qdrant_client

@step
def create_and_store_index(nodes_file: str):
    """Loads preprocessed nodes from file, creates an index, and stores embeddings in Qdrant."""
    # First check if the collection already has data
    if qdrant_client is not None:
        try:
            print(f"Checking if Qdrant collection '{COLLECTION_NAME}' already has data...")
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            if collection_info.points_count > 0:
                print(f"✅ Qdrant collection '{COLLECTION_NAME}' already has {collection_info.points_count} vectors.")
                print("Skipping index creation and data storage step.")
                return "index_already_exists"
            else:
                print(f"Collection exists but is empty ({collection_info.points_count} vectors). Will proceed with indexing.")
        except Exception as e:
            print(f"Error checking Qdrant collection: {str(e)}")
            print("Will attempt to create and store index anyway.")
    
    # Load nodes from the pickle file
    with open(nodes_file, "rb") as f:
        nodes = pickle.load(f)
    
    print(f"Loaded {len(nodes)} nodes from {nodes_file}, now creating index and storing in Qdrant...")
    
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
