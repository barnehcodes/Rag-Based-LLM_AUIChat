#!/usr/bin/env python3
"""
RAG Application with Qdrant Integration for AUIChat
Simplified for cloud deployment
"""

import logging
import os
import random
import pickle
from typing import Dict, Any, List
import re

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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def initialize_rag():
    """Initialize the RAG components including Qdrant client, embedding model, and vector index"""
    global qdrant_client_instance, vector_store, index, embed_model
    
    try:
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        embed_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        Settings.embed_model = embed_model
        
        # Connect to Qdrant
        logger.info("Connecting to Qdrant...")
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        
        # Log connection details (excluding sensitive information)
        logger.info(f"Qdrant URL: {qdrant_url}")
        logger.info(f"Using API key: {'Yes' if qdrant_api_key else 'No'}")
        
        # Initialize Qdrant client with API key if provided
        if qdrant_api_key:
            logger.info("Initializing Qdrant client with API key...")
            qdrant_client_instance = qdrant_client.QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            logger.info("Initializing Qdrant client without API key...")
            qdrant_client_instance = qdrant_client.QdrantClient(url=qdrant_url)
        
        # Create vector store and index
        logger.info("Creating vector store with Qdrant...")
        collection_name = os.environ.get("QDRANT_COLLECTION", "auichat_docs")
        logger.info(f"Using collection: {collection_name}")
        
        # First, check if we can load from the preprocessed nodes file
        try:
            logger.info("Attempting to load from preprocessed nodes...")
            with open("preprocessed_nodes.pkl", "rb") as f:
                nodes = pickle.load(f)
                
            logger.info(f"Loaded {len(nodes)} preprocessed nodes")
            
            vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=collection_name,
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load the index from the vector store
            logger.info("Loading index from vector store...")
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )
            
        except Exception as e:
            logger.error(f"Failed to load from preprocessed nodes: {e}")
            logger.info("Will initialize empty index...")
            vector_store = QdrantVectorStore(
                client=qdrant_client_instance,
                collection_name=collection_name,
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )
        
        # Initialize the LLM
        logger.info("Setting up LLM...")
        llm_model_name = os.environ.get("LLM_MODEL_NAME", "HuggingFaceTB/SmolLM-360M-Instruct") # Changed default
        logger.info(f"Using LLM model: {llm_model_name}")
        Settings.llm = HuggingFaceLLM(
            model_name=llm_model_name,
            tokenizer_name=llm_model_name,
            max_new_tokens=512,
            context_window=2048,
            generate_kwargs={"temperature": 0.2, "do_sample": True}, # Lowered temperature
        )
        
        logger.info("RAG components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
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
    global qdrant_client_instance, index
    
    qdrant_status = "ok"
    try:
        if qdrant_client_instance:
            # Simple test to check if Qdrant is responsive
            qdrant_client_instance.get_collections()
        else:
            qdrant_status = "not_initialized"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        qdrant_status = "error"
    
    return jsonify({
        "status": "ok",
        "qdrant": qdrant_status,
        "index_loaded": index is not None
    })

@app.route("/api/health")
def api_health():
    """API-prefixed health check endpoint"""
    return health()

@app.route("/predict", methods=["POST"])
def predict():
    """Legacy endpoint for compatibility with older clients"""
    return chat()

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
        response_obj = query_engine.query(query) # Renamed from response
        
        response_text = str(response_obj)

        # --- BEGIN ADDED FORMATTING ---
        # Format "Step X:" to be on a new line after a period.
        # e.g., ". Step 1:" becomes ".\nStep 1:"
        response_text = re.sub(r'\.\s*(Step\s+\d+:)', r'.\n\1', response_text)

        # Format "**Question X:" to "* Question X:"
        response_text = re.sub(r'\*\*(Question\s+\d+:)', r'* \1', response_text)
        # --- END ADDED FORMATTING ---

        # --- BEGIN ADDED LOGGING ---
        if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
            logger.info(f"Retrieved {len(response_obj.source_nodes)} source_nodes for query: '{query}'")
            for i, node_with_score in enumerate(response_obj.source_nodes):
                logger.info(f"Source Node {i+1} (Score: {node_with_score.score:.4f}):")
                logger.info(f"Content: {node_with_score.node.get_text()[:500]}...") 
                if hasattr(node_with_score.node, "metadata"):
                    logger.info(f"Metadata: {node_with_score.node.metadata}")
        else:
            logger.info(f"No source_nodes retrieved for query: '{query}'")
        # --- END ADDED LOGGING ---
        
        # Extract source information if available
        sources = []
        if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
            for node in response_obj.source_nodes:
                if hasattr(node, "metadata") and node.metadata:
                    source_info = {
                        "file_name": node.metadata.get("file_name", "Unknown"),
                        "score": round(node.score, 4) if hasattr(node, "score") else None,
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    }
                    sources.append(source_info)
        
        logger.info(f"Returning response with {len(sources)} sources")
        return jsonify({
            "response": response_text,  # Use the modified text
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
        response_obj = query_engine.query(query) # Renamed from response

        response_text = str(response_obj)

        # --- BEGIN ADDED FORMATTING ---
        # Format "Step X:" to be on a new line after a period.
        # e.g., ". Step 1:" becomes ".\nStep 1:"
        response_text = re.sub(r'\.\s*(Step\s+\d+:)', r'.\n\1', response_text)

        # Format "**Question X:" to "* Question X:"
        response_text = re.sub(r'\*\*(Question\s+\d+:)', r'* \1', response_text)
        # --- END ADDED FORMATTING ---

        # --- BEGIN ADDED LOGGING ---
        if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
            logger.info(f"Retrieved {len(response_obj.source_nodes)} source_nodes for query: '{query}'")
            for i, node_with_score in enumerate(response_obj.source_nodes):
                logger.info(f"Source Node {i+1} (Score: {node_with_score.score:.4f}):")
                logger.info(f"Content: {node_with_score.node.get_text()[:500]}...")
                if hasattr(node_with_score.node, "metadata"):
                    logger.info(f"Metadata: {node_with_score.node.metadata}")
        else:
            logger.info(f"No source_nodes retrieved for query: '{query}'")
        # --- END ADDED LOGGING ---
        
        sources = []
        if hasattr(response_obj, "source_nodes") and response_obj.source_nodes:
            for node in response_obj.source_nodes:
                if hasattr(node, "metadata") and node.metadata:
                    source_info = {
                        "file_name": node.metadata.get("file_name", "Unknown"),
                        "score": round(node.score, 4) if hasattr(node, "score") else None,
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    }
                    sources.append(source_info)
        
        logger.info(f"Returning response with {len(sources)} sources")
        return jsonify({
            "response": response_text,  # Use the modified text
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
    port = int(os.environ.get("PORT", 8080))
    
    # Try to initialize RAG components
    initialize_rag()
    
    logger.info(f"Starting AUIChat RAG API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)