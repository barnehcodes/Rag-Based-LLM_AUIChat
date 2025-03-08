from zenml import step
import json
from pipeline.config import qdrant_client, COLLECTION_NAME

@step
def validate_qdrant_storage():
    """Checks if embeddings are correctly stored in Qdrant."""
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Total vectors in '{COLLECTION_NAME}': {collection_info.points_count}")
    
    search_results, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=5,
        with_payload=True,
        with_vectors=True
    )
    
    if not search_results:
        print("🚨 No records retrieved from Qdrant!")
        return False
    
    print(f"✅ Storage validation successful!")
    return True
