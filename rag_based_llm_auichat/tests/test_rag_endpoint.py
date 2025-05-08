"""
Test script to check if the RAG endpoints are working properly.
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from zenml.logger import get_logger

logger = get_logger(__name__)

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent))

# Define a list of test queries
TEST_QUERIES = [
    "What are the admission requirements for transfer students?",
    
]

def test_rag_endpoint(endpoint_url=None, timeout=30):
    """
    Tests if a RAG endpoint is working properly by sending a list of test queries.
    
    Args:
        endpoint_url: URL of the endpoint to test. If None, tries to find one 
                     from the environment or info files.
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (bool, dict) indicating success and response information
    """
    # If no endpoint URL is provided, try to find one
    if endpoint_url is None:
        # Try different possible locations for endpoint URLs
        possible_info_files = [
            "/home/barneh/Rag-Based-LLM_AUIChat/cloudrun_qdrant_info.json",
            "/home/barneh/Rag-Based-LLM_AUIChat/cloudrun_optimized_info.json",
            "/home/barneh/Rag-Based-LLM_AUIChat/cloudrun_deployment_info.json"
        ]
        
        # First try environment variable
        endpoint_url = os.environ.get("AUICHAT_ENDPOINT_URL")
        if endpoint_url:
            logger.info(f"Using endpoint URL from environment variable: {endpoint_url}")
        else:
            # Try to load from info files
            for file_path in possible_info_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            info = json.load(f)
                            if 'service_url' in info:
                                endpoint_url = info['service_url']
                                logger.info(f"Using endpoint URL from {file_path}: {endpoint_url}")
                                break
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {str(e)}")
    
    if not endpoint_url:
        logger.error("No endpoint URL provided or found")
        return False, {"error": "No endpoint URL provided or found"}
    
    # Add /predict to the URL if it doesn't end with it
    if not endpoint_url.endswith('/predict'):
        endpoint_url = endpoint_url.rstrip('/') + '/predict'
    
    all_tests_passed = True
    results_summary = []
    
    for test_query in TEST_QUERIES:
        logger.info(f"Testing query: '{test_query}'")
        # Create request payload
        payload = {
            "instances": [
                {
                    "query": test_query,
                    "debug": True
                }
            ]
        }
        
        try:
            logger.info(f"Sending test query to {endpoint_url}: '{test_query}'")
            start_time = time.time()
            
            # Send the request
            response = requests.post(
                endpoint_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            request_time = time.time() - start_time
            logger.info(f"Request completed in {request_time:.2f} seconds with status code {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"Request failed for query '{test_query}' with status code {response.status_code}: {response.text}")
                all_tests_passed = False
                results_summary.append({
                    "query": test_query,
                    "status_code": response.status_code,
                    "error": f"Request failed: {response.text}",
                    "request_time": request_time,
                    "passed": False
                })
                continue
            
            # Parse the response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response for query '{test_query}': {response.text}")
                all_tests_passed = False
                results_summary.append({
                    "query": test_query,
                    "error": f"Invalid JSON response: {response.text}",
                    "passed": False
                })
                continue

            # Check if the response has predictions
            if "predictions" not in response_data or not isinstance(response_data["predictions"], list) or not response_data["predictions"]:
                logger.error(f"Response for query '{test_query}' doesn't contain valid predictions")
                all_tests_passed = False
                results_summary.append({
                    "query": test_query,
                    "error": "Response doesn't contain valid predictions",
                    "response": response_data,
                    "passed": False
                })
                continue
            
            prediction = response_data["predictions"][0]
            
            # Check if the prediction contains an answer
            if "answer" not in prediction or not prediction["answer"]:
                logger.error(f"Prediction for query '{test_query}' doesn't contain a valid answer")
                all_tests_passed = False
                results_summary.append({
                    "query": test_query,
                    "error": "Prediction doesn't contain a valid answer",
                    "prediction": prediction,
                    "passed": False
                })
                continue
            
            # Check if the prediction contains sources
            has_sources = "sources" in prediction and isinstance(prediction["sources"], list) and bool(prediction["sources"])
            if not has_sources:
                logger.warning(f"Prediction for query '{test_query}' doesn't contain sources or sources are empty.")
            
            # Check for debug info if requested
            search_method = "unknown"
            if "debug_info" in prediction and isinstance(prediction["debug_info"], dict):
                search_method = prediction["debug_info"].get("search_params", {}).get("method", "unknown")
                logger.info(f"Search method used for query '{test_query}': {search_method}")
            
            # Evaluate success for this query: must have an answer
            query_passed = bool(prediction.get("answer"))
            if not query_passed:
                all_tests_passed = False

            current_query_result = {
                "query": test_query,
                "endpoint": endpoint_url,
                "request_time": request_time,
                "has_answer": query_passed,
                "has_sources": has_sources,
                "search_method": search_method,
                "answer_preview": prediction.get("answer", "")[:100] + "..." if prediction.get("answer") else "",
                "sources_count": len(prediction.get("sources", []) if isinstance(prediction.get("sources"), list) else []),
                "status": "success" if query_passed else "failed",
                "passed": query_passed
            }
            results_summary.append(current_query_result)
            logger.info(f"Query '{test_query}' processed. Passed: {query_passed}")

        except requests.RequestException as e:
            logger.error(f"Request error for query '{test_query}': {str(e)}")
            all_tests_passed = False
            results_summary.append({"query": test_query, "error": f"Request error: {str(e)}", "passed": False})
        except Exception as e:
            logger.error(f"Unexpected error for query '{test_query}': {str(e)}")
            all_tests_passed = False
            results_summary.append({"query": test_query, "error": f"Unexpected error: {str(e)}", "passed": False})

    final_result_payload = {
        "overall_success": all_tests_passed,
        "individual_results": results_summary
    }
    
    if all_tests_passed:
        logger.info("All RAG endpoint checks passed.")
    else:
        logger.error("Some RAG endpoint checks failed.")
        
    return all_tests_passed, final_result_payload

if __name__ == "__main__":
    # Run the test directly if called as a script
    overall_success, summary_info = test_rag_endpoint()
    print(json.dumps(summary_info, indent=4))
    if overall_success:
        print("✅ Overall RAG endpoint check passed.")
        sys.exit(0)
    else:
        print("❌ Overall RAG endpoint check failed.")
        sys.exit(1)