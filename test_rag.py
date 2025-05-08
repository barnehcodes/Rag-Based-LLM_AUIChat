#!/usr/bin/env python3
"""
Simple test script for the AUIChat RAG model on Cloud Run
"""
import os
import requests
import json
import sys
import argparse

# List of known fallback responses from the RAG application
FALLBACK_RESPONSES = [
    "I apologize, but I'm having trouble retrieving that information at the moment. Could you try asking in a different way?",
    "I don't have enough information to answer that question properly. Could you provide more details or ask something else?",
    "I'm sorry, but I couldn't find reliable information to answer your question. Please try a different question or contact the university directly.",
    "That's a good question, but I'm not able to provide accurate information on that right now. Could we try a different topic?",
    "I'm still learning about Al Akhawayn University. I don't have enough context to answer that question properly yet."
]

def test_rag_query(query, endpoint_url=None, debug=False):
    """Test the RAG model with a query and verify LLM-generated answer with sources."""
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
    # Adjusted payload to match the server's expected format for /predict -> chat()
    payload = {
        "query": query
        # The server-side chat() function doesn't currently use a debug flag from the payload
        # to modify its response structure with detailed debug_info.
        # If you want to pass it, it would be: "debug_mode": debug
    }
    
    try:
        # Send the request
        response = requests.post(
            f"{url}/predict",
            headers=headers,
            json=payload,
            timeout=300
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        print(f"📥 Response received:")
        
        if "response" in result and isinstance(result["response"], str):
            answer = result["response"]
            sources = result.get("sources", []) # Optional: get sources if available
            
            print("📥 Response received:")
            print(f"   Answer: {answer}")
            if sources:
                print(f"   Sources ({len(sources)}):")
                for i, source in enumerate(sources):
                    print(f"     [{i+1}] Score: {source.get('score', 'N/A')}, ID: {source.get('id', 'N/A')}")
                    if debug and 'text' in source:
                        print(f"         Text: {source['text'][:100]}...") # Print first 100 chars
            else:
                print("   No sources provided in the response.")
            
            # Basic check for relevance (can be improved)
            if "admission requirements" in answer.lower() and "freshmen" in answer.lower():
                print("✅ Basic relevance check passed.")
            else:
                print("⚠️ Basic relevance check failed. Answer might not be relevant.")
                
        else: # 'response' key not found or not a string
            print(f"❌ Error: No 'response' key found or it's not a string in the result. Full response:\n{result}")
            
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON response. Status: {response.status_code}, Response text:\n{response.text}")
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
    success = test_rag_query(args.query, args.url, args.debug)
    print("=" * 50)
    if success:
        print("✅ Test Passed: RAG query successful with LLM-generated answer and sources.")
    else:
        print("❌ Test Failed: RAG query did not meet verification criteria (see details above).")
    sys.exit(0 if success else 1)
