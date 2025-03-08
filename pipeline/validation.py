# validation.py
import json
from config import qdrant_client, COLLECTION_NAME

def validate_qdrant_storage():
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
    
    print(f"Retrieved {len(search_results)} records for inspection")
    
    for result in search_results:
        print(f"\nRecord ID: {result.id}")
        print(f"Payload keys: {list(result.payload.keys())}")
        if hasattr(result, "vector") and result.vector is not None:
            print(f"✅ Vector dimensions: {len(result.vector)}")
        else:
            print("🚨 WARNING: Vector is None")
    
    print("✅ Storage validation successful!")
    return True
