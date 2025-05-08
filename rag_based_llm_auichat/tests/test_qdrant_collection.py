"""
Test script to check if the Qdrant collection is properly populated.
"""
import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from zenml.logger import get_logger

logger = get_logger(__name__)

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent))

def test_qdrant_collection(collection_name="AUIChatVectoreCol-384", min_vectors=500):
    """
    Tests if a Qdrant collection exists and has a minimum number of vectors.
    
    Args:
        collection_name: Name of the Qdrant collection to check
        min_vectors: Minimum number of vectors expected in the collection
        
    Returns:
        Tuple of (bool, dict) indicating success and collection information
    """
    try:
        # Qdrant connection details
        QDRANT_HOST = os.environ.get(
            "QDRANT_HOST",
            "40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"
        )
        QDRANT_API_KEY = os.environ.get(
            "QDRANT_API_KEY",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM"
        )

        logger.info(f"Connecting to Qdrant at {QDRANT_HOST} to check collection '{collection_name}'")
        
        # Connect to Qdrant
        client = QdrantClient(
            host=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
            https=True,
            timeout=20.0
        )
        
        # Get list of collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Check if our collection exists
        if collection_name not in collection_names:
            logger.error(f"Collection '{collection_name}' not found. Available collections: {collection_names}")
            return False, {"error": f"Collection '{collection_name}' not found", "available_collections": collection_names}
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else "unknown"
        
        logger.info(f"Collection '{collection_name}' has {points_count} vectors with dimension {vector_size}")
        
        # Check if the collection has the minimum number of vectors
        if points_count < min_vectors:
            logger.warning(f"Collection '{collection_name}' has only {points_count} vectors, expected at least {min_vectors}")
            return False, {
                "collection": collection_name,
                "points_count": points_count,
                "vector_size": vector_size,
                "expected_min_vectors": min_vectors,
                "status": "insufficient_vectors"
            }
        
        # Sample a few points to make sure they have proper content
        try:
            sample_points, _ = client.scroll(
                collection_name=collection_name,
                limit=2,
                with_payload=True
            )
            
            if not sample_points:
                logger.warning(f"Collection '{collection_name}' has no points with payload")
                return False, {"error": "Collection has no points with payload"}
            
            # Check if payload contains content
            has_content = False
            content_fields = ['_node_content', 'text', 'content', 'page_content']
            
            for point in sample_points:
                for field in content_fields:
                    if field in point.payload and point.payload[field]:
                        has_content = True
                        break
            
            if not has_content:
                logger.warning(f"Sample points do not contain expected content fields: {content_fields}")
                return False, {"error": f"Sample points do not contain expected content fields: {content_fields}"}
                
            logger.info(f"Sample points have expected content. Collection '{collection_name}' is valid.")
            return True, {
                "collection": collection_name,
                "points_count": points_count,
                "vector_size": vector_size,
                "status": "valid"
            }
            
        except Exception as e:
            logger.error(f"Error checking sample points: {str(e)}")
            return False, {"error": f"Error checking sample points: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {str(e)}")
        return False, {"error": f"Error connecting to Qdrant: {str(e)}"}

if __name__ == "__main__":
    # Run the test directly if called as a script
    success, info = test_qdrant_collection()
    if success:
        print(f"✅ Qdrant collection check passed: {info}")
        sys.exit(0)
    else:
        print(f"❌ Qdrant collection check failed: {info}")
        sys.exit(1)