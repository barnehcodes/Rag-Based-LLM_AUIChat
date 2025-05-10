#!/usr/bin/env python3
"""
Simple test script for the AUIChat RAG model on Cloud Run
"""
import os
import requests
import json
import sys
import argparse

def test_rag_query(query, endpoint_url=None, debug=False):
    """Test the RAG model with a query"""
    # Get endpoint URL from environment variable if not provided
    url = endpoint_url or os.environ.get("AUICHAT_ENDPOINT_URL")
    
    if not url:
        print("❌ Error: No endpoint URL provided")
        print("Please set the AUICHAT_ENDPOINT_URL environment variable or provide --url parameter")
        return False
    
    print(f"🔍 Testing RAG query on endpoint: {url}")
    print(f"   Query: '{query}'")
    
    # Prepare the request to the /predict endpoint
    headers = {"Content-Type": "application/json"}
    payload = {
        "instances": [
            {"query": query, "debug": debug}
        ]
    }
    
    try:
        # Send the request
        response = requests.post(
            f"{url}/predict",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        print(f"📥 Response received:")
        
        # Print the answer
        if "predictions" in result and len(result["predictions"]) > 0:
            prediction = result["predictions"][0]
            
            if isinstance(prediction, dict) and "answer" in prediction:
                print("\n=== Answer ===")
                print(prediction["answer"])
                
                if "sources" in prediction:
                    print("\n=== Sources ===")
                    for i, source in enumerate(prediction["sources"]):
                        print(f"\nSource {i+1}:")
                        # Print first 300 chars of each source
                        print(f"{source[:300]}..." if len(source) > 300 else source)
                
                # Display debug info if available
                if debug and "debug_info" in prediction:
                    print("\n=== Debug Info ===")
                    debug_info = prediction["debug_info"]
                    
                    # Print top chunks and their similarity scores
                    if "top_chunks" in debug_info:
                        print("\nTop retrieved chunks and similarity scores:")
                        for i, chunk_info in enumerate(debug_info["top_chunks"]):
                            print(f"\nChunk {i+1} (Score: {chunk_info['score']:.4f}):")
                            print(f"{chunk_info['text'][:200]}...")
                    
                    # Print search parameters
                    if "search_params" in debug_info:
                        print("\nSearch parameters:")
                        for param, value in debug_info["search_params"].items():
                            print(f"- {param}: {value}")
                    
                    # Print timing information
                    if "timings" in debug_info:
                        print("\nTiming information:")
                        for step, time_ms in debug_info["timings"].items():
                            print(f"- {step}: {time_ms:.2f}ms")
            else:
                print(f"Prediction: {prediction}")
        else:
            print("No predictions found in the response")
            
        return True
    
    except Exception as e:
        print(f"❌ Error testing RAG model: {str(e)}")
        if hasattr(e, "response") and e.response:
            print(f"Response content: {e.response.text}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the AUIChat RAG model")
    parser.add_argument("--query", type=str, 
                      default="What are the admission requirements for freshmen?",
                      help="The question to ask the RAG model")
    parser.add_argument("--url", type=str,
                      help="The Cloud Run service URL (optional if AUICHAT_ENDPOINT_URL is set)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode to see more information about the retrieval process")
    
    args = parser.parse_args()
    
    # Run the test
    print("🧪 AUIChat RAG Model Testing Tool")
    print("=" * 50)
    test_rag_query(args.query, args.url, args.debug)
