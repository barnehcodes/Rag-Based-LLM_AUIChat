#!/usr/bin/env python3
"""
Minimal CORS Proxy for AUIChat
"""

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# RAG service endpoint
RAG_ENDPOINT = os.environ.get('RAG_ENDPOINT', 'https://auichat-rag-qdrant-h4ikwiq3ja-uc.a.run.app')

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({"status": "ok", "service": "CORS Proxy"})

@app.route('/api/health')
def health():
    """Health check endpoint that forwards to the RAG endpoint"""
    try:
        # Don't actually call the RAG endpoint to avoid any potential timeouts
        return jsonify({
            "status": "ok",
            "proxy_status": "online",
            "rag_endpoint": RAG_ENDPOINT
        })
    except Exception as e:
        print(f"Error in health check: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Forward chat requests to the RAG API"""
    if request.method == 'OPTIONS':
        # Handle preflight requests
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        return response
    
    try:
        response = requests.post(
            f"{RAG_ENDPOINT}/api/chat",
            json=request.json,
            timeout=30
        )
        # Return response with CORS headers
        return Response(
            response.content,
            status=response.status_code,
            headers={
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        )
    except Exception as e:
        print(f"Error proxying request: {e}")
        return jsonify({
            "response": "Sorry, I couldn't connect to the RAG service.",
            "sources": []
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting CORS proxy on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)