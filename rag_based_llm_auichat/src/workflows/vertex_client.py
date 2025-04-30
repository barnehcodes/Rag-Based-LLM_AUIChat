"""
Vertex AI client module for AUIChat
Provides utilities to interact with models deployed on Vertex AI endpoints
"""
from typing import Dict, Any, List, Optional
import requests
import json
import os
import google.auth
import google.auth.transport.requests
from zenml.logger import get_logger

logger = get_logger(__name__)

class VertexEndpointClient:
    """Client to interact with models deployed to Vertex AI endpoints"""
    
    def __init__(
        self, 
        endpoint_url: str, 
        project_id: Optional[str] = None,
        region: str = "us-central1"
    ):
        """
        Initialize the Vertex AI endpoint client
        
        Args:
            endpoint_url: Full URL to the prediction endpoint
            project_id: GCP project ID (uses default if None)
            region: GCP region for deployment
        """
        self.endpoint_url = endpoint_url
        self.project_id = project_id
        self.region = region
        self._token = None
        self._token_expiry = 0
        
    def _get_auth_token(self) -> str:
        """Get a valid authentication token for GCP API calls"""
        # Check if we have a valid cached token
        import time
        current_time = int(time.time())
        
        if self._token and current_time < self._token_expiry - 300:  # 5-minute buffer
            return self._token
            
        try:
            # Get credentials and token
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            self._token = credentials.token
            self._token_expiry = credentials.expiry
            
            return self._token
        except Exception as e:
            logger.error(f"Failed to get authentication token: {e}")
            raise RuntimeError(f"Authentication error: {str(e)}")
    
    def predict(self, text: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Make a prediction request to the deployed model
        
        Args:
            text: Input text for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary with prediction results
        """
        # Get authentication token
        token = self._get_auth_token()
        
        # Prepare headers with authentication
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the request payload
        # Note: The exact format may vary depending on your model type
        payload = {
            "instances": [
                {
                    "prompt": text,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            ]
        }
        
        try:
            # Make the prediction request
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse and return the response
            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Prediction request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response content: {e.response.text}")
            raise RuntimeError(f"Prediction request failed: {str(e)}")

def get_vertex_client_for_deployment(deployment_info: Dict[str, Any]) -> VertexEndpointClient:
    """
    Creates a Vertex client for a specific deployment
    
    Args:
        deployment_info: The deployment information from deploy_model_to_vertex step
        
    Returns:
        A configured VertexEndpointClient
    """
    endpoint_url = deployment_info.get("endpoint_url")
    project_id = deployment_info.get("project_id")
    region = deployment_info.get("region")
    
    if not endpoint_url:
        raise ValueError("Missing endpoint_url in deployment_info")
        
    return VertexEndpointClient(
        endpoint_url=endpoint_url,
        project_id=project_id,
        region=region
    )