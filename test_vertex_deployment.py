#!/bin/bash
# test_vertex_deployment.py - Test script for verifying Vertex AI deployment

import os
import argparse
import json
from rag_based_llm_auichat.src.workflows.vertex_client import VertexEndpointClient

def test_deployment(endpoint_url, project_id, region, prompt="What is AUIChat?"):
    """Test a deployed Vertex AI endpoint with a simple prompt"""
    print(f"🔍 Testing Vertex AI endpoint: {endpoint_url}")
    print(f"📝 Using prompt: '{prompt}'")
    
    # Create the client
    client = VertexEndpointClient(
        endpoint_url=endpoint_url,
        project_id=project_id,
        region=region
    )
    
    try:
        # Make a prediction
        print("📨 Sending request to endpoint...")
        response = client.predict(
            text=prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        # Print formatted response
        print("\n✅ Received response:")
        print(json.dumps(response, indent=2))
        
        return True
    except Exception as e:
        print(f"❌ Error testing endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Vertex AI deployment")
    parser.add_argument("--endpoint-url", required=True, help="The full endpoint URL")
    parser.add_argument("--project-id", default="deft-waters-458118-a3", help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--prompt", default="What is AUIChat?", help="Test prompt to send")
    
    args = parser.parse_args()
    test_deployment(args.endpoint_url, args.project_id, args.region, args.prompt)