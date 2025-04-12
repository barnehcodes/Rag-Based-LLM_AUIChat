#!/usr/bin/env python3
from flask import Flask, request, Response
import requests

app = Flask(__name__)

# Target API
TARGET_API = 'http://localhost:5000'

@app.route('/<path:path>', methods=['GET', 'POST', 'OPTIONS'])
def proxy(path):
    # Handle OPTIONS preflight requests
    if request.method == 'OPTIONS':
        response = Response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    # Forward the request to the target API
    url = f"{TARGET_API}/{path}"
    
    # Print for debugging
    print(f"Forwarding {request.method} request to {url}")
    print(f"Request data: {request.get_data()}")
    
    try:
        # Forward the request with the same method and headers
        resp = requests.request(
            method=request.method,
            url=url,
            headers={key: value for (key, value) in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False
        )
        
        # Create a Flask response object
        response = Response(resp.content, resp.status_code)
        
        # Add CORS headers to the response
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        # Forward headers from the target API's response
        for name, value in resp.headers.items():
            if name.lower() != 'access-control-allow-origin':  # Don't copy existing CORS headers
                response.headers[name] = value
                
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error forwarding request: {e}")
        return Response(f"Error connecting to target API: {e}", status=500)

if __name__ == '__main__':
    # Run on port 5001
    print("🚀 Starting CORS Proxy on port 5001...")
    print("Forwarding requests to http://localhost:5000")
    print("Access your API via http://localhost:5001/api/chat")
    app.run(host='0.0.0.0', port=5001, debug=True)