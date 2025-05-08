"""
Test script to check if the embedding model for vector similarity search is working properly.
"""
import os
import sys
from pathlib import Path
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from zenml.logger import get_logger

logger = get_logger(__name__)

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent))

def test_embedding_model(model_name="BAAI/bge-small-en-v1.5"):
    """
    Tests if the embedding model is working properly by generating and comparing embeddings.
    
    Args:
        model_name: Name of the embedding model to test
        
    Returns:
        Tuple of (bool, dict) indicating success and information about the model
    """
    try:
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        
        # Try to load the model
        model = SentenceTransformer(model_name)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Get embedding dimension
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model embedding dimension: {embedding_dim}")
        
        # Test with a few queries
        test_queries = [
            "What are the admission requirements for transfer students?",
            "How can I apply for financial aid?",
            "When is the application deadline?"
        ]
        
        logger.info(f"Generating embeddings for {len(test_queries)} test queries")
        start_time = time.time()
        
        # Generate embeddings for all test queries
        embeddings = model.encode(test_queries)
        
        encode_time = time.time() - start_time
        logger.info(f"Embeddings generated in {encode_time:.2f} seconds")
        
        # Check if embeddings have the expected shape
        if embeddings.shape != (len(test_queries), embedding_dim):
            logger.error(f"Embedding shape {embeddings.shape} doesn't match expected shape {(len(test_queries), embedding_dim)}")
            return False, {"error": f"Unexpected embedding shape: {embeddings.shape}"}
        
        # Test vector similarity by comparing embeddings
        similarity_matrix = np.zeros((len(test_queries), len(test_queries)))
        
        for i in range(len(test_queries)):
            for j in range(len(test_queries)):
                # Compute cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i, j] = similarity
        
        logger.info("Similarity matrix between test queries:")
        for i in range(len(test_queries)):
            logger.info(f"  {test_queries[i][:30]}...: {similarity_matrix[i]}")
            
        # Check if self-similarity is high (should be close to 1.0)
        for i in range(len(test_queries)):
            if similarity_matrix[i, i] < 0.99:
                logger.error(f"Self-similarity for query {i} is too low: {similarity_matrix[i, i]}")
                return False, {"error": f"Self-similarity check failed: {similarity_matrix[i, i]} < 0.99"}
                
        # Check if similarity between different queries is less than self-similarity
        for i in range(len(test_queries)):
            for j in range(len(test_queries)):
                if i != j and similarity_matrix[i, j] >= similarity_matrix[i, i]:
                    logger.error(f"Similarity between query {i} and {j} is higher than self-similarity")
                    return False, {"error": f"Similarity check failed: {similarity_matrix[i, j]} >= {similarity_matrix[i, i]}"}
                    
        # All checks passed
        return True, {
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "load_time": load_time,
            "encode_time": encode_time,
            "similarity_matrix": similarity_matrix.tolist(),
            "status": "valid"
        }
        
    except Exception as e:
        logger.error(f"Error testing embedding model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False, {"error": f"Error testing embedding model: {str(e)}"}
        
if __name__ == "__main__":
    # Run the test directly if called as a script
    # Try both embedding models we might be using
    for model_name in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
        print(f"\nTesting embedding model: {model_name}")
        success, info = test_embedding_model(model_name)
        if success:
            print(f"✅ Embedding model check passed: {model_name}")
        else:
            print(f"❌ Embedding model check failed for {model_name}: {info}")