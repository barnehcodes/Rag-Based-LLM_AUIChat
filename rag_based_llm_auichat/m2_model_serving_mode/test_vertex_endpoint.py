#!/usr/bin/env python3
"""
Test script for Vertex AI endpoint
"""
import os
import json
import argparse
import requests
from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Cloud Run Adapter (added by deployment script)
# This adapter allows the test script to work with the Cloud Run endpoint
# ---------------------------------------------------------------------------

# Environment variable for the Cloud Run endpoint URL
AUICHAT_ENDPOINT_URL = os.environ.get('AUICHAT_ENDPOINT_URL', 'https://auichat-rag-service-02e9ac41-h4ikwiq3ja-uc.a.run.app')

# Mock Vertex AI client for Cloud Run
class CloudRunVertexAdapter:
    def init(self, project_id=None, location=None):
        pass
        
    class Endpoint:
        def __init__(self, endpoint_name=None):
            self.cloud_run_url = AUICHAT_ENDPOINT_URL
            
        def predict(self, instances):
            headers = {"Content-Type": "application/json"}
            payload = {"instances": instances}
            
            response = requests.post(
                f"{self.cloud_run_url}/predict",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"{response.status_code} {response.text}")
                
            result = response.json()
            
            # Create a response object that mimics Vertex AI
            class MockResponse:
                def __init__(self, predictions, deployed_model_id):
                    self.predictions = predictions
                    self.deployed_model_id = deployed_model_id
            
            return MockResponse(
                predictions=result.get("predictions", []),
                deployed_model_id=result.get("deployed_model_id", "auichat-rag-cloudrun")
            )
            
# Replace the real aiplatform with our adapter
import sys
from types import ModuleType
cloud_run_vertex = CloudRunVertexAdapter()
sys.modules['google.cloud.aiplatform'] = cloud_run_vertex

# ---------------------------------------------------------------------------
# End of Cloud Run Adapter
# ---------------------------------------------------------------------------
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# Set up argument parsing
parser = argparse.ArgumentParser(description='Test a Vertex AI endpoint with sample queries')
parser.add_argument('--endpoint-id', type=str, 
                    help='The Vertex AI endpoint ID to test (e.g., 6124144526782103552)')
parser.add_argument('--project-id', type=str, default=os.environ.get('PROJECT_ID', 'deft-waters-458118-a3'),
                    help='Google Cloud Project ID')
parser.add_argument('--region', type=str, default=os.environ.get('REGION', 'us-central1'),
                    help='Google Cloud Region')
args = parser.parse_args()

# Check if endpoint ID was provided or use the default from the recent deployment
ENDPOINT_ID = args.endpoint_id or "6124144526782103552"
PROJECT_ID = args.project_id
REGION = args.region

def test_endpoint():
    """Test the Vertex AI endpoint with a sample question"""
    print(f"🔍 Testing Vertex AI endpoint: {ENDPOINT_ID}")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Region: {REGION}")
    
    # Initialize Vertex AI client
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Get the endpoint
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
    
    # For a linear regression model (just testing connectivity)
    test_instance = [[1.0, 2.0]]
    
    print(f"📤 Sending test prediction request: {test_instance}")
    
    try:
        # Send prediction request
        response = endpoint.predict(instances=test_instance)
        
        # Print response
        print(f"📥 Received response:")
        print(f"   Prediction: {response.predictions}")
        print(f"   Deployed model ID: {response.deployed_model_id}")
        
        print("✅ Endpoint test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing endpoint: {str(e)}")
        return False

def test_rag_query(query="What are the admission requirements for freshmen?"):
    """
    Test the RAG model with a query about AUI admissions
    """
    print(f"🔍 Testing RAG query on endpoint: {ENDPOINT_ID}")
    print(f"   Query: '{query}'")
    
    # Initialize Vertex AI client
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Get the endpoint
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
    
    # Format the query as expected by your model
    # This depends on your model's input format, adjust as needed
    instance = {
        "query": query
    }
    
    print(f"📤 Sending RAG query")
    
    try:
        # Send prediction request - we'll try different formats since we're not sure
        # of the exact format your endpoint expects
        try:
            # Try as a JSON structure
            response = endpoint.predict(instances=[instance])
        except Exception as e1:
            try:
                # Try as a raw string
                response = endpoint.predict(instances=[query])
            except Exception as e2:
                # Try as a Value protobuf
                value = Value()
                value.struct_value.update({"query": query})
                response = endpoint.predict(instances=[value])
        
        # Print response
        print(f"📥 Received response:")
        print(f"   {response.predictions}")
        
        print("✅ RAG query test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG query: {str(e)}")
        print("Note: This may be expected if your model isn't actually a RAG model")
        return False

if __name__ == "__main__":
    print("🧪 Vertex AI Endpoint Testing Tool")
    print("=" * 50)
    
    # First test basic connectivity with the simple model we deployed
    success = test_endpoint()
    
    if success:
        # Since we know the deployed model is a simple sklearn model,
        # the RAG test will likely fail, but we'll try anyway for completeness
        print("\nAttempting RAG query test (may fail with simple model):")
        test_rag_query()
        
        print("\n📋 Next Steps:")
        print("1. Deploy your actual AUIChat RAG model instead of the test model")
        print("2. Use the UI components in your project to interact with the model")
        print("3. For programmatic access, adjust this script to match your model's input/output format")
        print("\n💡 Endpoint URL in Google Cloud Console:")
        print(f"https://console.cloud.google.com/vertex-ai/endpoints/{ENDPOINT_ID}?project={PROJECT_ID}")
    else:
        print("\n❌ Basic endpoint test failed. Please check:")
        print("1. That the endpoint ID is correct")
        print("2. Your Google Cloud authentication is set up")
        print("3. The endpoint is still active in the Vertex AI console")