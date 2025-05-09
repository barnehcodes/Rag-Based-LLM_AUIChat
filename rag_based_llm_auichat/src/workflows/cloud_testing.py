from zenml import step
from zenml.logger import get_logger
import requests
import json
from typing import Dict, Any

logger = get_logger(__name__)

@step
def test_cloud_run_endpoint_step(cloud_run_deployment_info: Dict[str, Any]) -> str:
    """
    Tests the deployed Cloud Run endpoint by sending a POST request with a test query.
    
    Args:
        cloud_run_deployment_info: Dictionary containing deployment info, including service_url
        
    Returns:
        Status message indicating whether the test passed or failed
    """
    service_url = cloud_run_deployment_info.get("service_url")
    if not service_url:
        logger.error("Service URL not found in Cloud Run deployment info")
        return "test_failed_missing_service_url"
    
    # The endpoint URL for predictions
    predict_endpoint = f"{service_url}/predict"
    logger.info(f"Testing endpoint: {predict_endpoint}")
    
    # Test query
    test_query = "What are the admission requirements for freshmen?"
    
    # Payload formatted for the improved_rag_app_qdrant.py API
    payload = {
        "query": test_query
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        logger.info(f"Sending test query: '{test_query}'")
        response = requests.post(predict_endpoint, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for error status codes
        
        # Parse the response
        response_data = response.json()
        logger.info(f"Received response with status code: {response.status_code}")
        
        # Check if the response has expected structure
        if "response" in response_data:
            logger.info("Test passed: Received response with expected structure")
            logger.info(f"Response excerpt: {response_data['response'][:100]}...")
            return "test_passed_received_valid_response"
        else:
            logger.warning("Test completed, but response format was unexpected")
            logger.warning(f"Received keys: {list(response_data.keys())}")
            return "test_completed_unexpected_response_format"
            
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error during test: {http_err}")
        if hasattr(http_err, 'response') and http_err.response is not None:
            logger.error(f"Response status code: {http_err.response.status_code}")
            logger.error(f"Response text: {http_err.response.text}")
        return f"test_failed_http_error_{getattr(http_err.response, 'status_code', 'unknown')}"
        
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error during test: {conn_err}")
        return "test_failed_connection_error"
        
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error during test: {timeout_err}")
        return "test_failed_timeout"
        
    except Exception as e:
        logger.error(f"Unexpected error during test: {e}")
        return f"test_failed_unexpected_error: {str(e)}"