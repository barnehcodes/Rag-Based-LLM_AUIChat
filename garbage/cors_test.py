"""
Simple CORS test server to verify CORS functionality
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes and origins
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with CORS headers"""
    return jsonify({
        "status": "healthy",
        "message": "CORS test service is ready"
    })

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint with CORS headers"""
    return jsonify({
        "message": "CORS is working correctly!",
        "service": "cors-test"
    })

if __name__ == "__main__":
    # Run the Flask server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)