#!/usr/bin/env python
"""
CORS Proxy Service for AUIChat
This service proxies requests from the UI to the RAG endpoint and adds CORS headers to the responses.
Deploy this alongside your UI on Cloud Run to fix CORS issues.
"""

from flask import Flask, request, Response, jsonify
import requests
from flask_cors import CORS
import os
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes with explicit wildcard

# Get RAG endpoint from environment variable or use default
RAG_ENDPOINT = os.environ.get('RAG_ENDPOINT', 'https://auichat-rag-qdrant-h4ikwiq3ja-uc.a.run.app')
logger.info(f"Configured with RAG endpoint: {RAG_ENDPOINT}")

@app.route('/')
def root():
    """Root endpoint - simple status check"""
    return jsonify({
        "status": "ok",
        "service": "CORS Proxy for AUIChat RAG",
        "rag_endpoint": RAG_ENDPOINT
    })

@app.route('/health')
def health():
    """Health check endpoint that forwards to the RAG endpoint's health check"""
    try:
        logger.info(f"Checking health of RAG endpoint at {RAG_ENDPOINT}/health")
        response = requests.get(f"{RAG_ENDPOINT}/health", timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.RequestException as e:
        logger.error(f"Error checking RAG endpoint health: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Could not connect to RAG endpoint",
            "qdrant": "unavailable",
            "proxy_status": "ok"
        }), 200  # Return 200 to indicate the proxy is working even if backend isn't

@app.route('/chat', methods=['POST'])
def chat():
    """Proxy chat requests to the RAG endpoint"""
    try:
        logger.info("Received chat request")
        # Forward the request to the RAG endpoint
        response = requests.post(
            f"{RAG_ENDPOINT}/chat", 
            json=request.json,
            headers={k: v for k, v in request.headers if k.lower() != 'host'},
            timeout=30
        )
        logger.info(f"RAG endpoint responded with status code {response.status_code}")
        return Response(
            response.content,
            status=response.status_code,
            headers={'Content-Type': response.headers.get('Content-Type', 'application/json'),
                    'Access-Control-Allow-Origin': '*'}
        )
    except requests.RequestException as e:
        logger.error(f"Error communicating with RAG endpoint: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error communicating with RAG service",
            "details": str(e)
        }), 200  # Return 200 with error message for better client handling

@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy_api(path):
    """Generic API proxy for any endpoint"""
    # Handle OPTIONS requests (preflight) directly
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.update({
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        })
        return response
    
    try:
        logger.info(f"Proxying request to {RAG_ENDPOINT}/{path}")
        
        # Forward the request with the same method, headers, and body
        response = requests.request(
            method=request.method,
            url=f"{RAG_ENDPOINT}/{path}",
            headers={k: v for k, v in request.headers.items() if k.lower() != 'host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=30
        )
        
        # Create a Flask response with the same content and status code
        proxy_response = Response(
            response.content,
            status=response.status_code
        )
        
        # Add headers from the original response
        for key, value in response.headers.items():
            if key.lower() not in ('transfer-encoding', 'connection'):
                proxy_response.headers[key] = value
        
        # Ensure CORS headers are present
        proxy_response.headers['Access-Control-Allow-Origin'] = '*'
                
        return proxy_response
        
    except requests.RequestException as e:
        logger.error(f"Error proxying to RAG endpoint: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return jsonify({
            "status": "error",
            "message": "Error communicating with RAG service",
            "details": str(e)
        }), 200  # Return 200 with error message

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 8080))
        logger.info(f"Starting CORS proxy service on port {port}")
        logger.info(f"Proxying requests to RAG endpoint: {RAG_ENDPOINT}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        sys.exit(1)