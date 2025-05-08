"""
Cloud Run RAG Client for AUIChat
Provides a client to interact with the deployed RAG model on Cloud Run
"""
from typing import Dict, Any, List, Optional, Union
import requests
import os
import json
from urllib.parse import urljoin

class CloudRunRagClient:
    """Client to interact with the RAG model deployed on Cloud Run"""
    
    def __init__(self, endpoint_url: Optional[str] = None):
        """
        Initialize the Cloud Run RAG client
        
        Args:
            endpoint_url: The URL for the Cloud Run service endpoint
                          If None, uses the AUICHAT_ENDPOINT_URL environment variable
        """
        self.endpoint_url = endpoint_url or os.environ.get("AUICHAT_ENDPOINT_URL")
        if not self.endpoint_url:
            raise ValueError(
                "Endpoint URL not provided and AUICHAT_ENDPOINT_URL environment variable not set"
            )
        
        # Ensure the endpoint URL has the correct format
        if not self.endpoint_url.startswith("http"):
            self.endpoint_url = f"https://{self.endpoint_url}"
            
        # Ensure the URL ends with a slash for reliable joins
        if not self.endpoint_url.endswith("/"):
            self.endpoint_url += "/"
            
        print(f"Initialized CloudRunRagClient with endpoint: {self.endpoint_url}")
        
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the RAG service
        
        Returns:
            Status information from the service
        """
        try:
            response = requests.get(
                urljoin(self.endpoint_url, "health"),
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": f"Health check failed: {str(e)}"}
    
    def query(self, text: str) -> Dict[str, Any]:
        """
        Query the RAG model with a question
        
        Args:
            text: The query text to send to the RAG model
            
        Returns:
            Dictionary with response data including answer and sources
        """
        try:
            # Format the request as expected by the RAG service
            payload = {
                "instances": [
                    {"query": text}
                ]
            }
            
            # Send the query to the prediction endpoint
            response = requests.post(
                urljoin(self.endpoint_url, "predict"),
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse and return the response
            result = response.json()
            
            # Extract the relevant part of the response
            predictions = result.get("predictions", [])
            if predictions and len(predictions) > 0:
                return predictions[0]
            else:
                return {"answer": "No answer found", "sources": []}
            
        except Exception as e:
            return {"error": f"Query failed: {str(e)}"}

def get_rag_client() -> CloudRunRagClient:
    """
    Factory function to get a configured RAG client
    
    Returns:
        Configured CloudRunRagClient instance
    """
    # Default to the environment variable, or use the most recently deployed service
    endpoint_url = os.environ.get("AUICHAT_ENDPOINT_URL")
    
    # If not found, try to load from deployment info files
    if not endpoint_url:
        try:
            # Try the optimized deployment first
            with open("/home/barneh/Rag-Based-LLM_AUIChat/cloudrun_optimized_info.json", "r") as f:
                info = json.load(f)
                endpoint_url = info.get("service_url")
        except FileNotFoundError:
            try:
                # Fall back to the original deployment
                with open("/home/barneh/Rag-Based-LLM_AUIChat/cloudrun_deployment_info.json", "r") as f:
                    info = json.load(f)
                    endpoint_url = info.get("service_url")
            except FileNotFoundError:
                # No URL information found
                endpoint_url = None
    
    return CloudRunRagClient(endpoint_url)

if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    client = get_rag_client()
    
    print("Testing RAG service health...")
    health = client.health_check()
    print(f"Health check result: {health}")
    
    print("\nTesting RAG query...")
    query_text = "What are the admission requirements for freshmen?"
    result = client.query(query_text)
    
    print(f"Query: '{query_text}'")
    print("\nResponse:")
    
    if "answer" in result:
        print("\n=== Answer ===")
        print(result["answer"])
        print("\n=== Sources ===")
        for i, source in enumerate(result.get("sources", [])):
            print(f"\nSource {i+1}:")
            print(source[:200] + "..." if len(source) > 200 else source)
    else:
        print(result)
