#!/usr/bin/env python3
"""
RAG Application with Qdrant Integration for AUIChat
Simplified for cloud deployment
"""

import logging
import os
import random
import pickle
import re # Added import
import traceback # Added import
from typing import Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS
import qdrant_client
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Added name to format
)
logger = logging.getLogger(__name__)
logger.info("SCRIPT_START: improved_rag_app_qdrant.py is starting.") # New log

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Fallback responses when RAG retrieval fails
FALLBACK_RESPONSES = [
    "I apologize, but I'm having trouble retrieving that information at the moment. Could you try asking in a different way?",
    "I don't have enough information to answer that question properly. Could you provide more details or ask something else?",
    "I'm sorry, but I couldn't find reliable information to answer your question. Please try a different question or contact the university directly.",
    "That's a good question, but I'm not able to provide accurate information on that right now. Could we try a different topic?",
    "I'm still learning about Al Akhawayn University. I don't have enough context to answer that question properly yet."
]

# Initialize global variables for the RAG components
qdrant_client_instance = None
vector_store = None
index = None
embed_model = None
llm_model = None # Added for the local LLM
nodes = None # Initialize nodes as a global variable

def initialize_rag():
    """Initialize the RAG components including Qdrant client, embedding model, and vector index"""
    global qdrant_client_instance, vector_store, index, embed_model, llm_model, nodes # Added nodes
    
    logger.info("INITIALIZE_RAG_START: Attempting to initialize RAG components.") # New log

    # Set HuggingFace cache directory - should align with Dockerfile's HF_HOME
    # This ensures the app uses the pre-baked models from the image
    cache_dir = os.environ.get("HF_HOME", "/app/huggingface_cache")
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
    logger.info(f"ENV_HF_HOME: {os.environ.get('HF_HOME')}") # New log
    logger.info(f"Application using HuggingFace cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        logger.warning(f"Cache directory {cache_dir} does not exist. Models might be downloaded at runtime.")

    try:
        # Initialize embedding model - BAAI/bge-small-en-v1.5
        embed_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5") # Use env var, default to bge
        logger.info(f"Initializing embedding model {embed_model_name} from cache: {cache_dir}...")
        try:
            embed_model = HuggingFaceEmbedding(model_name=embed_model_name, cache_folder=cache_dir)
            Settings.embed_model = embed_model
            logger.info(f"EMBED_MODEL_SUCCESS: Embedding model {embed_model_name} initialized.") # New log
        except Exception as e:
            logger.error(f"EMBED_MODEL_FAILURE: Failed to initialize embedding model {embed_model_name}: {e}", exc_info=True) # New log with exc_info
            raise # Re-raise to be caught by the outer try-except

        # Connect to Qdrant
        logger.info("Connecting to Qdrant...")
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        collection_name = os.environ.get("QDRANT_COLLECTION", "AUIChatVectoreCol-384")

        logger.info(f"ENV_QDRANT_URL: {qdrant_url}") # New log
        logger.info(f"ENV_QDRANT_API_KEY_IS_SET: {'Yes' if qdrant_api_key else 'No'}") # New log - don't log the key itself
        logger.info(f"ENV_QDRANT_COLLECTION: {collection_name}") # New log

        if not qdrant_url:
            logger.error("QDRANT_URL environment variable not set.")
            return False
        
        try:
            logger.info(f"Attempting to connect to Qdrant at {qdrant_url} with collection {collection_name}")
            if qdrant_api_key:
                qdrant_client_instance = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                logger.warning("QDRANT_API_KEY environment variable not set. Proceeding without API key.")
                qdrant_client_instance = qdrant_client.QdrantClient(url=qdrant_url)
            # Test connection by listing collections (or a more lightweight operation if available)
            qdrant_client_instance.get_collections()
            logger.info("QDRANT_CLIENT_SUCCESS: Qdrant client initialized and connection tested.") # New log
        except Exception as e:
            logger.error(f"QDRANT_CLIENT_FAILURE: Failed to initialize or connect Qdrant client: {e}", exc_info=True) # New log
            return False # Return False as this is critical

        # Create vector store and index
        logger.info("Creating vector store with Qdrant...")
        
        nodes_loaded_from_pickle = False
        try:
            logger.info("Attempting to load from preprocessed_nodes.pkl...")
            with open("preprocessed_nodes.pkl", "rb") as f:
                nodes = pickle.load(f)
            logger.info(f"Loaded {len(nodes)} preprocessed nodes from preprocessed_nodes.pkl")
            nodes_loaded_from_pickle = True
        except FileNotFoundError:
            logger.warning("preprocessed_nodes.pkl not found. Will proceed without preloaded nodes if Qdrant collection is populated.")
        except Exception as e:
            logger.error(f"Failed to load preprocessed_nodes.pkl: {e}", exc_info=True)
            # Decide if this is critical. If nodes are essential for index creation from scratch, then it might be.
            # For now, we assume the index can still be loaded from an existing Qdrant collection.

        try:
            vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            logger.info("Loading index from vector store...")
            # If nodes were loaded, LlamaIndex might use them if the vector store is empty or being built.
            # If the vector store is already populated in Qdrant, it will load from there.
            if nodes_loaded_from_pickle and nodes: # Check if nodes is not empty
                 # This path might be taken if you intend to populate Qdrant from these nodes
                 # However, typically, you load an *existing* index from Qdrant.
                 # If the goal is to load an index *already in Qdrant*, nodes are not directly used here.
                 # For clarity, let's assume we are loading an existing index.
                 # If the collection is new/empty and nodes are meant to populate it, the logic would be different (e.g. VectorStoreIndex.from_documents)
                logger.info("Attempting to load index from vector store (preprocessed nodes are available but typically not used for loading an *existing* index directly like this).")
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    # embed_model=embed_model # embed_model is already in Settings
                )
            else:
                logger.info("Attempting to load index from an existing Qdrant vector store (no local preprocessed nodes to insert).")
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                    # embed_model=embed_model
                )
            logger.info("VECTOR_STORE_INDEX_SUCCESS: Vector store and index initialized/loaded.") # New log
        except Exception as e:
            logger.error(f"VECTOR_STORE_INDEX_FAILURE: Failed to initialize vector store or index: {e}", exc_info=True) # New log
            return False # Return False as this is critical
        
        # Initialize the LLM
        llm_model_name = os.environ.get("LLM_MODEL_NAME", "HuggingFaceTB/SmolLM-360M-Instruct") 
        logger.info(f"ENV_LLM_MODEL_NAME: {llm_model_name}") # New log
        logger.info(f"Initializing LLM: {llm_model_name} from cache: {cache_dir}...")
        try:
            Settings.llm = HuggingFaceLLM(
                model_name=llm_model_name, 
                tokenizer_name=llm_model_name,
                max_new_tokens=256,
                context_window=1024, # Increased from 512
                generate_kwargs={"temperature": 0.7, "do_sample": True},
                # cache_folder is not a direct param for HuggingFaceLLM, it respects HF_HOME
            )
            llm_model = Settings.llm # Assign to global
            logger.info(f"LLM_MODEL_SUCCESS: LLM {llm_model_name} initialized.") # New log
        except Exception as e:
            logger.error(f"LLM_MODEL_FAILURE: Failed to initialize LLM {llm_model_name}: {e}", exc_info=True) # New log
            return False # Return False as this is critical
        
        logger.info("INITIALIZE_RAG_SUCCESS: RAG components initialized successfully.") # New log
        return True
    except Exception as e:
        logger.error(f"INITIALIZE_RAG_FAILURE: Unhandled exception during RAG component initialization: {e}", exc_info=True) # New log
        return False

@app.route("/")
def root():
    """Root endpoint returning basic status information"""
    return jsonify({
        "status": "ok",
        "service": "AUIChat RAG API",
        "version": "1.0"
    })

@app.route("/health")
def health():
    """Health check endpoint that also reports on Qdrant connection"""
    global qdrant_client_instance, index, embed_model, llm_model
    
    qdrant_status = "not_initialized"
    qdrant_collection_name = os.environ.get("QDRANT_COLLECTION", "AUIChatVectoreCol-384")
    qdrant_points_count = None
    try:
        if qdrant_client_instance:
            # Simple test to check if Qdrant is responsive
            qdrant_client_instance.get_collections() # Check connection
            qdrant_status = "ok"
            try:
                collection_info = qdrant_client_instance.get_collection(collection_name=qdrant_collection_name)
                qdrant_points_count = collection_info.points_count
            except Exception as e:
                logger.warning(f"Could not get info for collection {qdrant_collection_name}: {e}")
                qdrant_status = "collection_error"

        else:
            qdrant_status = "client_not_initialized"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        qdrant_status = "error"
    
    return jsonify({
        "status": "ok",
        "qdrant_status": qdrant_status,
        "qdrant_collection": qdrant_collection_name,
        "qdrant_points_count": qdrant_points_count,
        "index_loaded": index is not None,
        "embedding_model_loaded": embed_model is not None,
        "llm_model_loaded": llm_model is not None,
        "embedding_model_name": Settings.embed_model.model_name if Settings.embed_model else None,
        "llm_model_name": Settings.llm.model_name if Settings.llm else None,
    })

@app.route("/api/health")
def api_health():
    """API-prefixed health check endpoint"""
    return health()

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the /predict endpoint for generating answers based on user queries."""
    global index, llm_model, nodes # Ensure llm_model and nodes are accessible
    logger.info("PREDICT_ENDPOINT_REQUEST_RECEIVED: Received request for /predict.") # New log

    if index is None or llm_model is None: # Check if LLM is also initialized
        logger.error("PREDICT_ENDPOINT_ERROR: RAG components (index or LLM) not initialized.") # Updated log
        # Attempt to re-initialize if not already initialized
        if not initialize_rag():
            logger.error("PREDICT_ENDPOINT_ERROR: Failed to re-initialize RAG components during predict.")
            return jsonify({
                "error": "RAG components are not initialized, and re-initialization failed. Please check server logs."
            }), 500
        # If re-initialization was successful, update global references (though initialize_rag should do this)
        # This might be redundant if initialize_rag correctly sets globals, but good for safety.
        # from llama_index.core import Settings # Local import if needed
        # llm_model = Settings.llm


    try:
        data = request.get_json()
        if not data or "query" not in data:
            logger.warning("PREDICT_ENDPOINT_INVALID_REQUEST: Invalid request data. 'query' field missing.") # New log
            return jsonify({"error": "Invalid request. 'query' field is required."}), 400
        
        query_text = data["query"]
        logger.info(f"PREDICT_ENDPOINT_QUERY: Received query: '{query_text}'") # New log

        # Initialize query engine if RAG components are ready
        # Ensure llm_model is passed if it's a global or part of Settings
        if index and llm_model: # Check both index and llm_model
            logger.info("Initializing query engine...")
            try:
                # Use the globally set Settings.llm which should be configured by initialize_rag
                query_engine = index.as_query_engine(
                    similarity_top_k=3, 
                    response_mode="compact" # Using compact for potentially more concise answers
                    # llm=llm_model # Not needed if Settings.llm is correctly set
                )
                logger.info("QUERY_ENGINE_SUCCESS: Query engine initialized.") # New log
            except Exception as e:
                logger.error(f"QUERY_ENGINE_FAILURE: Failed to initialize query engine: {e}", exc_info=True) # New log
                return jsonify({"error": f"Failed to initialize query engine: {e}"}), 500
        else:
            logger.error("PREDICT_ENDPOINT_ERROR: Index or LLM not available for query engine.") # Updated log
            return jsonify({"error": "RAG components (index or LLM) not ready for querying."}), 500

        logger.info(f"Querying with text: {query_text}")
        response = query_engine.query(query_text)
        answer = str(response)
        
        # Log the raw answer before cleaning
        logger.info(f"PREDICT_ENDPOINT_RAW_ANSWER: Raw answer from RAG: '{answer}'")


        # Clean the answer: Remove control characters that might cause issues in JSON or display
        # This is the line that caused the error.
        # Original: cleaned_answer = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', answer)
        # Corrected pattern to ensure it's interpreted as ASCII and avoid potential hidden char issues.
        # Using a more robust way to define the character set for removal.
        # The original pattern r'[\x00-\x1f\x7f-\x9f]' is generally correct.
        # If it fails, it might be due to non-ASCII characters in the pattern string itself in the .py file.
        # We ensure it's a raw string and uses standard escapes.
        control_chars_pattern = r"[\x00-\x1f\x7f-\x9f]"
        cleaned_answer = re.sub(control_chars_pattern, '', answer)
        
        # Log the cleaned answer
        logger.info(f"PREDICT_ENDPOINT_CLEANED_ANSWER: Cleaned answer: '{cleaned_answer}'")


        # Fallback mechanism if the answer is empty or seems unhelpful
        if not cleaned_answer or cleaned_answer.strip() == "" or "I don't have enough information" in cleaned_answer or "I cannot answer" in cleaned_answer: # Added more checks
            logger.warning(f"PREDICT_ENDPOINT_FALLBACK: RAG returned an empty or unhelpful answer. Using fallback. Original: '{cleaned_answer}'") # New log
            cleaned_answer = random.choice(FALLBACK_RESPONSES)
        
        # Log the final answer being sent
        logger.info(f"PREDICT_ENDPOINT_FINAL_RESPONSE: Sending response: '{cleaned_answer}'") # New log
        return jsonify({"response": cleaned_answer, "sources": [{"id": node.node_id, "text": node.text} for node in response.source_nodes]})

    except Exception as e:
        tb_str = traceback.format_exc() # Get traceback string
        logger.error(f"PREDICT_ENDPOINT_UNHANDLED_EXCEPTION: An error occurred during prediction: {e}\\nTraceback:\\n{tb_str}") # New log with traceback
        # Provide a generic fallback if any other error occurs
        return jsonify({"answer": random.choice(FALLBACK_RESPONSES), "error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Process an old-style chat request using RAG"""
    global index
    
    try:
        data = request.json
        
        # Try to extract the query from different possible formats
        query = None
        
        if data and "query" in data:
            query = data["query"]
        elif data and "prompt" in data:
            query = data["prompt"]
        else:
            return jsonify({"error": "No query found in request"}), 400
        
        logger.info(f"Processing query: {query}")
        
        # If the index is not initialized, try to initialize it
        if index is None:
            success = initialize_rag()
            if not success:
                logger.warning("Using fallback response due to RAG initialization failure")
                return jsonify({
                    "response": random.choice(FALLBACK_RESPONSES),
                    "sources": []
                })
        
        # Create query engine and execute query
        query_engine = index.as_query_engine(similarity_top_k=3)
        response_obj = query_engine.query(query)
        
        # Extract source information if available
        sources = []
        if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
            for node in response_obj.source_nodes:
                if hasattr(node, "metadata") and node.metadata:
                    source_info = {
                        "file_name": node.metadata.get("file_name", "Unknown"),
                        "score": round(node.score, 4) if hasattr(node, "score") else None,
                        "text": node.text[:300] + "..." if len(node.text) > 300 else node.text # Increased preview length
                    }
                    sources.append(source_info)
        
        response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)

        # Clean up the response text from unwanted metadata patterns
        # Example pattern: "**Q: 255**\nPage_label: 5\nfile_path: /path/to/file.pdf\n\n"
        # This regex will find such blocks. Using re.escape on parts of the pattern is safer if they could contain regex special characters.
        # However, for this specific known pattern, direct regex is fine.
        import re
        # Regex to find and remove the metadata blocks. It looks for:
        # - Optional leading newlines
        # - "**Q: <digits>**" (Q: part is bolded in markdown)
        # - "Page_label: <digits>"
        # - "file_path: <any characters until newline>"
        # - Followed by one or more newlines
        cleaned_response_text = re.sub(r'\n*\*\*Q: \d+\*\*\nPage_label: \d+\nfile_path: [^\n]+\n*\n*', '', response_text).strip()
        # A simpler regex if the Q: part is not always bold or formatting varies:
        # cleaned_response_text = re.sub(r'\n*Q: \d+\nPage_label: \d+\nfile_path: [^\n]+\n*\n*', '', response_text, flags=re.IGNORECASE).strip()

        logger.info(f"Returning response with {len(sources)} sources. Cleaned answer: '{cleaned_response_text[:100]}...'")
        return jsonify({
            "response": cleaned_response_text,
            "sources": sources
        })
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({
            "response": random.choice(FALLBACK_RESPONSES),
            "sources": []
        })

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    """Process a chat request using RAG with the /api prefix"""
    global index
    
    # Handle preflight requests for CORS
    if request.method == "OPTIONS":
        response = jsonify({"message": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request format"}), 400
        
        # Handle both formats: messages array and direct query
        query = None
        
        if "messages" in data:
            # Extract the last user message from messages array
            messages = data["messages"]
            last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            
            if not last_user_message or "content" not in last_user_message:
                return jsonify({"error": "No valid user message found in messages"}), 400
            
            query = last_user_message["content"]
        elif "query" in data:
            query = data["query"]
        elif "prompt" in data:
            query = data["prompt"]
        else:
            return jsonify({"error": "No query or messages found in request"}), 400
        
        logger.info(f"Processing query: {query}")
        
        # If the index is not initialized, try to initialize it
        if index is None:
            success = initialize_rag()
            if not success:
                logger.warning("Using fallback response due to RAG initialization failure")
                return jsonify({
                    "response": random.choice(FALLBACK_RESPONSES),
                    "sources": []
                })
        
        # Create query engine and execute query
        query_engine = index.as_query_engine(similarity_top_k=3)
        response_obj = query_engine.query(query)
        
        # Extract source information if available
        sources = []
        if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
            for node in response_obj.source_nodes:
                if hasattr(node, "metadata") and node.metadata:
                    source_info = {
                        "file_name": node.metadata.get("file_name", "Unknown"),
                        "score": round(node.score, 4) if hasattr(node, "score") else None,
                        "text": node.text[:300] + "..." if len(node.text) > 300 else node.text # Increased preview length
                    }
                    sources.append(source_info)
        
        response_text = response_obj.response if hasattr(response_obj, 'response') else str(response_obj)

        # Clean up the response text from unwanted metadata patterns
        # Example pattern: "**Q: 255**\nPage_label: 5\nfile_path: /path/to/file.pdf\n\n"
        # This regex will find such blocks. Using re.escape on parts of the pattern is safer if they could contain regex special characters.
        # However, for this specific known pattern, direct regex is fine.
        import re
        # Regex to find and remove the metadata blocks. It looks for:
        # - Optional leading newlines
        # - "**Q: <digits>**" (Q: part is bolded in markdown)
        # - "Page_label: <digits>"
        # - "file_path: <any characters until newline>"
        # - Followed by one or more newlines
        cleaned_response_text = re.sub(r'\n*\*\*Q: \d+\*\*\nPage_label: \d+\nfile_path: [^\n]+\n*\n*', '', response_text).strip()
        # A simpler regex if the Q: part is not always bold or formatting varies:
        # cleaned_response_text = re.sub(r'\n*Q: \d+\nPage_label: \d+\nfile_path: [^\n]+\n*\n*', '', response_text, flags=re.IGNORECASE).strip()

        logger.info(f"Returning response with {len(sources)} sources. Cleaned answer: '{cleaned_response_text[:100]}...'")
        return jsonify({
            "response": cleaned_response_text,
            "sources": sources
        })
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({
            "response": random.choice(FALLBACK_RESPONSES),
            "sources": []
        })

# Initialize RAG components on startup
if __name__ == "__main__":
    logger.info("MAIN_BLOCK_START: Entering main execution block.") # New log
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"PORT: {port}") # New log
    
    # Try to initialize RAG components
    logger.info("MAIN_BLOCK_CALL_INIT_RAG: Calling initialize_rag().") # New log
    initialization_successful = initialize_rag()
    
    if initialization_successful:
        logger.info("MAIN_BLOCK_INIT_RAG_SUCCESS: initialize_rag() completed successfully.") # New log
    else:
        logger.error("MAIN_BLOCK_INIT_RAG_FAILURE: initialize_rag() failed. The application might not function correctly.") # New log
    
    logger.info(f"Starting AUIChat RAG API on port {port}")
    # Note: Flask's built-in server is not recommended for production.
    # Gunicorn is typically used in Cloud Run (specified in Procfile or entrypoint).
    # If using Flask's server directly, ensure debug=False for production.
    app.run(host="0.0.0.0", port=port, debug=False)
    logger.info("MAIN_BLOCK_END: Flask app.run() has exited (should not happen in normal Cloud Run operation).") # New log