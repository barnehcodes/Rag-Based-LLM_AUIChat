"""
Development testing step for AUIChat
This step runs a series of tests to ensure the development environment is properly set up
"""
from zenml import step
from zenml.logger import get_logger
import os
import sys
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the tests
from tests.test_qdrant_collection import test_qdrant_collection
from tests.test_rag_endpoint import test_rag_endpoint
from tests.test_embedding_model import test_embedding_model

logger = get_logger(__name__)

@step
def run_development_tests(
    collection_name: str = "AUIChatVectoreCol-384",
    endpoint_url: Optional[str] = None,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",  # Changed default
    min_vectors: int = 500
) -> Dict[str, Any]:
    """
    Run a series of tests to validate the development environment is working correctly.
    
    Args:
        collection_name: Name of the Qdrant collection to check
        endpoint_url: URL of the RAG endpoint to test (if None, tries to find one)
        embedding_model_name: Name of the embedding model to test
        min_vectors: Minimum number of vectors expected in the collection
        
    Returns:
        Dictionary with test results
    """
    test_results = {
        "all_passed": True,
        "tests": {}
    }
    
    # Step 1: Test Qdrant collection
    logger.info("====================================")
    logger.info("STEP 1: Testing Qdrant collection...")
    logger.info("====================================")
    
    qdrant_success, qdrant_info = test_qdrant_collection(
        collection_name=collection_name,
        min_vectors=min_vectors
    )
    
    test_results["tests"]["qdrant_collection"] = {
        "success": qdrant_success,
        "info": qdrant_info
    }
    
    if not qdrant_success:
        test_results["all_passed"] = False
        logger.error(f"❌ Qdrant collection test failed: {qdrant_info}")
        logger.error("This is a critical failure. Other tests may not work correctly.")
    else:
        logger.info(f"✅ Qdrant collection test passed: {qdrant_info}")
    
    # Step 2: Test embedding model
    logger.info("\n====================================")
    logger.info("STEP 2: Testing embedding model...")
    logger.info("====================================")
    
    embedding_success, embedding_info = test_embedding_model(
        model_name=embedding_model_name
    )
    
    test_results["tests"]["embedding_model"] = {
        "success": embedding_success,
        "info": embedding_info
    }
    
    if not embedding_success:
        test_results["all_passed"] = False
        logger.error(f"❌ Embedding model test failed: {embedding_info}")
        logger.error("This is a critical failure. Vector similarity search won't work correctly.")
    else:
        logger.info(f"✅ Embedding model test passed: {embedding_info}")
    
    # Step 3: Test RAG endpoint
    logger.info("\n====================================")
    logger.info("STEP 3: Testing RAG endpoint...")
    logger.info("====================================")
    
    endpoint_success = True
    endpoint_info = {}
    
    if qdrant_success:  # Only test endpoint if Qdrant is working
        endpoint_success, endpoint_info = test_rag_endpoint(
            endpoint_url=endpoint_url
        )
        
        test_results["tests"]["rag_endpoint"] = {
            "success": endpoint_success,
            "info": endpoint_info
        }
        
        if not endpoint_success:
            test_results["all_passed"] = False
            logger.error(f"❌ RAG endpoint test failed: {endpoint_info}")
            logger.error("The RAG service may not be deployed or is not functioning correctly.")
        else:
            logger.info(f"✅ RAG endpoint test passed: {endpoint_info}")
    else:
        logger.warning("Skipping RAG endpoint test because Qdrant collection test failed.")
        test_results["tests"]["rag_endpoint"] = {
            "success": False,
            "info": {"error": "Skipped because Qdrant collection test failed."}
        }
        test_results["all_passed"] = False
    
    # Summary
    logger.info("\n====================================")
    logger.info("DEVELOPMENT TEST SUMMARY")
    logger.info("====================================")
    
    tests_passed = sum(1 for test in test_results["tests"].values() if test["success"])
    total_tests = len(test_results["tests"])
    
    logger.info(f"Tests passed: {tests_passed} / {total_tests}")
    
    if test_results["all_passed"]:
        logger.info("✅ All tests passed! Development environment is ready.")
    else:
        logger.warning("⚠️ Some tests failed. Check the detailed results above.")
    
    return test_results