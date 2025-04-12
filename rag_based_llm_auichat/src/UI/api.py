from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import sys
import os
from pathlib import Path

# Add the src directory to the path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent
sys.path.append(str(src_dir))
sys.path.append(str(project_root))

# Import RAG components
from src.engines.query_engine import create_query_engine
from src.workflows.config import load_environment

# Initialize environment
try:
    load_environment()
    print("✅ Environment loaded successfully")
except Exception as e:
    print(f"❌ Error loading environment: {str(e)}")

app = Flask(__name__)

# Configure CORS with more specific settings
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Add explicit CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Handle OPTIONS requests explicitly
@app.route('/api/chat', methods=['OPTIONS'])
def options():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
    return response

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return RAG-based responses"""
    # Log incoming request for debugging
    print(f"Received request: {request.method} {request.path}")
    print(f"Request headers: {request.headers}")
    
    try:
        data = request.json
        user_query = data.get('message', '')
        
        if not user_query.strip():
            return jsonify({'response': 'Please enter a question.'}), 400
        
        # Process through the RAG pipeline
        try:
            start_time = __import__('time').time()
            
            # Instead of using the ZenML step directly, use llama_index directly
            from src.workflows.config import qdrant_client, COLLECTION_NAME, embed_model
            from llama_index.core import VectorStoreIndex
            from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            
            # Initialize the LLM
            llm = HuggingFaceInferenceAPI(
                model_name="mistralai/Mistral-7B-Instruct-v0.3", 
                token="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"
            )
            
            # Set up vector store
            temp_vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                text_key="text",
                metadata_key="metadata",
                content_key="content",
                embed_dim=768,
                stores_text=True
            )
            
            # Create index from vector store
            index = VectorStoreIndex.from_vector_store(temp_vector_store)
            
            # Create query engine
            query_engine = index.as_query_engine(
                llm=llm,
                similarity_top_k=3  # Retrieve top 3 most similar chunks
            )
            
            # Execute query
            response = query_engine.query(user_query)
            
            inference_time = (__import__('time').time() - start_time) * 1000  # Convert to ms
            
            return jsonify({
                'response': str(response),
                'metrics': {
                    'inferenceTime': round(inference_time, 2)
                }
            })
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'response': f"I encountered an error while processing your question. Please try again or contact support.\nError details: {str(e)}",
                'metrics': {
                    'inferenceTime': 0
                }
            }), 500
    except Exception as e:
        print(f"Error parsing request: {str(e)}")
        return jsonify({
            'response': f"Error parsing request: {str(e)}",
            'metrics': {
                'inferenceTime': 0
            }
        }), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'auichat-rag-api'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting AUIChat RAG API server on port {port}...")
    print(f"API endpoints available at:")
    print(f"  - http://localhost:{port}/api/chat (POST)")
    print(f"  - http://localhost:{port}/api/health (GET)")
    app.run(host='0.0.0.0', port=port, debug=True)