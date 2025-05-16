#!/bin/bash
# Script to deploy Version B of the RAG System to Cloud Run for A/B testing
# Based on successful patterns from deploy_rag_backend_cloudrun.sh

set -e

# Determine project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Function to validate prerequisites
validate_prerequisites() {
  local errors=0
  
  # Check if preprocessed nodes file exists
  local nodes_path="${PROJECT_ROOT}/rag_based_llm_auichat/preprocessed_nodes.pkl"
  if [[ ! -f "$nodes_path" ]]; then
    echo "❌ ERROR: Preprocessed nodes file not found at ${nodes_path}"
    echo "   Please ensure this file exists before deployment."
    errors=$((errors+1))
  else
    echo "✅ Preprocessed nodes file found at ${nodes_path}"
  fi
  
  # Check if gcloud is installed
  if ! command -v gcloud &> /dev/null; then
    echo "❌ ERROR: gcloud CLI not found"
    echo "   Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    errors=$((errors+1))
  else
    echo "✅ gcloud CLI found: $(which gcloud)"
  fi
  
  # Check if docker is installed
  if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: docker not found"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    errors=$((errors+1))
  else
    echo "✅ docker found: $(which docker)"
  fi
  
  # Check if authenticated with gcloud
  if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "❌ ERROR: Not authenticated with gcloud"
    echo "   Please run: gcloud auth login"
    errors=$((errors+1))
  else
    local account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    echo "✅ Authenticated with gcloud as: ${account}"
  fi
  
  # Check if specified project exists
  if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
    echo "❌ ERROR: Project $PROJECT_ID not found or not accessible"
    errors=$((errors+1))
  else
    echo "✅ GCP project found: ${PROJECT_ID}"
  fi
  
  if [[ $errors -gt 0 ]]; then
    echo ""
    echo "⚠️ Found ${errors} issue(s) that need to be resolved before deployment."
    return 1
  else
    echo ""
    echo "✅ All prerequisites passed. Ready to deploy."
    return 0
  fi
}

# Configuration (can be overridden with environment variables)
PROJECT_ID=${PROJECT_ID:-"deft-waters-458118-a3"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME=${SERVICE_NAME:-"auichat-rag-version-b"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
DRY_RUN=${DRY_RUN:-false} # Set to 'true' for testing without deployment

echo "🚀 Deploying Version B of AUIChat RAG with advanced features to Cloud Run"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Check if this is just a validation run
if [[ "$1" == "--validate-only" ]]; then
  echo "🔍 Validation mode: Only checking prerequisites, will not deploy"
  validate_prerequisites
  exit $?
fi

# Validate prerequisites before proceeding
echo "🔍 Validating deployment prerequisites..."
if ! validate_prerequisites; then
  echo "❌ Prerequisite check failed. Please resolve the issues and try again."
  exit 1
fi

# Create a temporary directory
TMP_DIR=$(mktemp -d)
echo "📁 Created temporary directory: ${TMP_DIR}"

# Cleanup function
cleanup() {
  echo "🧹 Cleaning up temporary directory"
  rm -rf "${TMP_DIR}"
}

# Register cleanup on exit
trap cleanup EXIT

# Copy necessary files
echo "📂 Preparing application files"
cp "${PROJECT_ROOT}/rag_based_llm_auichat/preprocessed_nodes.pkl" "${TMP_DIR}/"

# Create the enhanced Version B application code
cat > "${TMP_DIR}/app.py" << 'EOF'
#!/usr/bin/env python3
"""
Enhanced RAG Application (Version B) with advanced features:
1. Hybrid Retrieval (vector + BM25)
2. Query Reformulation
3. Advanced Re-ranking with cross-encoders
4. Multi-hop RAG architecture
5. Improved prompting
6. Better generation parameters
"""

import os
import sys
import time
import json
import torch
import logging
import pickle
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import random
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle different import structures for llama_index
try:
    # Try importing from llama_index.core (newer modular structure)
    from llama_index.core import VectorStoreIndex, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import BM25Retriever, VectorIndexRetriever, HybridRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import CompactAndRefine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    logger.info("Using llama-index modular structure imports")
except ImportError:
    try:
        # Try importing from llama_index directly (older structure)
        from llama_index import VectorStoreIndex, Settings, StorageContext
        from llama_index.node_parser import SentenceSplitter
        from llama_index.retrievers import BM25Retriever, VectorIndexRetriever, HybridRetriever
        from llama_index.query_engine import RetrieverQueryEngine
        from llama_index.response_synthesizers import CompactAndRefine
        from llama_index.postprocessor import SimilarityPostprocessor
        from llama_index.postprocessor.types import BaseNodePostprocessor
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        logger.info("Using llama-index direct imports")
    except ImportError:
        # Last resort: try importing specifically from llama_index_core
        from llama_index_core import VectorStoreIndex, Settings, StorageContext
        from llama_index_core.node_parser import SentenceSplitter
        from llama_index_core.retrievers import BM25Retriever, VectorIndexRetriever, HybridRetriever
        from llama_index_core.query_engine import RetrieverQueryEngine
        from llama_index_core.response_synthesizers import CompactAndRefine
        from llama_index_core.postprocessor import SimilarityPostprocessor
        from llama_index_core.postprocessor.types import BaseNodePostprocessor
        # These are likely still in their regular locations
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        logger.info("Using llama_index_core package imports")

# Import from sentence_transformers
from sentence_transformers import CrossEncoder

# Import transformers for query reformulation
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -------------------------------------------------
# Version B Enhanced Components
# -------------------------------------------------

class QueryReformulator:
    """
    Uses a T5-based model to reformulate user queries to improve retrieval.
    Expands acronyms, adds specific terms, and makes queries more precise.
    """
    def __init__(self):
        logger.info("Initializing query reformulation model...")
        model_name = "google/flan-t5-small"  # Smaller model for query reformulation
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("Query reformulation model loaded.")
        
        # Add specific context about AUI
        self.context_prefix = "Al Akhawayn University is a university in Ifrane, Morocco. "
        
        # Common acronyms in AUI context
        self.acronyms = {
            "AUI": "Al Akhawayn University in Ifrane",
            "SBA": "School of Business Administration",
            "SSE": "School of Science and Engineering",
            "SHSS": "School of Humanities and Social Sciences",
            "PiP": "Partners in Progress"
        }
        
    def expand_acronyms(self, query: str) -> str:
        """Expand known acronyms in the query"""
        expanded = query
        for acronym, expansion in self.acronyms.items():
            # Only replace if the acronym appears as a whole word
            expanded = expanded.replace(f" {acronym} ", f" {expansion} ")
            if expanded.startswith(f"{acronym} "):
                expanded = f"{expansion} " + expanded[len(acronym)+1:]
            if expanded.endswith(f" {acronym}"):
                expanded = expanded[:-len(acronym)-1] + f" {expansion}"
            if expanded == acronym:
                expanded = expansion
        return expanded
    
    def reformulate(self, query: str) -> str:
        """
        Reformulate the query to make it more specific and retrievable.
        """
        # Step 1: Expand any acronyms in the query
        expanded_query = self.expand_acronyms(query)
        
        # Step 2: Prepare prompt for the model
        prompt = f"Rewrite this question to make it more specific and detailed for a university information retrieval system. Original question: {expanded_query}"
        
        # Step 3: Generate reformulation
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=150,
            temperature=0.2,
            num_return_sequences=1
        )
        reformulated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If the model output is too different or doesn't make sense, use the expanded query
        if len(reformulated.split()) < 3 or len(reformulated) / len(expanded_query) < 0.5:
            logger.warning(f"Query reformulation failed, using expanded query: {expanded_query}")
            return expanded_query
        
        logger.info(f"Original: '{query}' → Reformulated: '{reformulated}'")
        return reformulated

class CrossEncoderReranker(BaseNodePostprocessor):
    """
    Uses a cross-encoder model to rerank retrieved documents for better precision.
    """
    def __init__(self, top_n: int = 5):
        logger.info("Initializing cross-encoder reranker...")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.top_n = top_n
        logger.info("Cross-encoder reranker initialized.")
    
    def postprocess_nodes(self, nodes, query_str):
        if not nodes:
            return []
        
        # Create pairs of (query, document text) for scoring
        pairs = [(query_str, node.get_text()) for node in nodes]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Assign new scores to nodes
        for node, score in zip(nodes, scores):
            node.score = float(score)  # Update node score with cross-encoder score
        
        # Sort by new score and take top_n
        reranked_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)[:self.top_n]
        
        return reranked_nodes

class MultiHopQueryEngine:
    """
    Implements a multi-hop RAG architecture for complex questions.
    First retrieval gets basic information, which is used for a second retrieval step.
    """
    def __init__(self, retriever, llm, similarity_top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        
        # Synthesizer for compact and more efficient responses
        self.response_synthesizer = CompactAndRefine(
            llm=self.llm,
            verbose=True,
        )
        
        # Standard query engine for first hop
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
        )
    
    def _needs_second_hop(self, query: str, first_hop_nodes) -> bool:
        """Determine if we need a second hop based on first results"""
        # Simple heuristic: if question is long or contains specific keywords
        complexity_indicators = ["specific", "details", "explain", "how", "requirements", 
                                "what are the", "tell me about", "prerequisites"]
        
        query_lower = query.lower()
        is_complex = (
            len(query.split()) >= 8 or
            any(indicator in query_lower for indicator in complexity_indicators)
        )
        
        # If we don't have enough context from first hop
        has_insufficient_context = len(first_hop_nodes) < 2
        
        return is_complex and not has_insufficient_context
    
    def _generate_follow_up_query(self, query: str, first_hop_response: str) -> str:
        """Generate a follow-up query based on the first response"""
        prompt = f"""
        Based on the following question and initial information, generate a follow-up question to get more specific details:
        
        Original Question: {query}
        
        Initial Information: {first_hop_response}
        
        Follow-up Question:
        """
        
        # Get the response
        response = self.llm.complete(prompt)
        follow_up = response.text.strip()
        
        if not follow_up:
            follow_up = f"Tell me more specific details about {query}"
            
        logger.info(f"Follow-up query: {follow_up}")
        return follow_up
    
    def query(self, query_str: str) -> tuple:
        """
        Execute multi-hop RAG query process.
        Returns both the response and nodes used for generation.
        """
        # First hop retrieval and response generation
        first_hop_nodes = self.retriever.retrieve(query_str)
        
        if not self._needs_second_hop(query_str, first_hop_nodes):
            # For simple queries, just use standard query engine
            response = self.query_engine.query(query_str)
            return response, first_hop_nodes
        
        # Generate intermediate answer from first hop
        first_response = self.response_synthesizer.synthesize(
            query=query_str,
            nodes=first_hop_nodes
        )
        
        # Generate follow-up query based on first response
        follow_up_query = self._generate_follow_up_query(
            query_str, 
            first_response.response
        )
        
        # Second hop retrieval
        second_hop_nodes = self.retriever.retrieve(follow_up_query)
        
        # Combine unique nodes from both hops
        all_node_ids = set()
        combined_nodes = []
        
        for node in first_hop_nodes + second_hop_nodes:
            if node.node_id not in all_node_ids:
                all_node_ids.add(node.node_id)
                combined_nodes.append(node)
        
        # Final response generation with all collected evidence
        final_response = self.response_synthesizer.synthesize(
            query=query_str,
            nodes=combined_nodes
        )
        
        return final_response, combined_nodes

# Fallback responses when errors occur
FALLBACK_RESPONSES = [
    "I apologize, but I'm having trouble retrieving that information at the moment. Could you try asking in a different way?",
    "I don't have enough information to answer that question properly. Could you provide more details or ask something else?",
    "I'm sorry, but I couldn't find reliable information to answer your question. Please try a different question or contact the university directly.",
    "That's a good question, but I'm not able to provide accurate information on that right now. Could we try a different topic?",
    "I'm still learning about Al Akhawayn University. I don't have enough context to answer that question properly yet."
]

# -------------------------------------------------
# Main RAG Application
# -------------------------------------------------

# Initialize FastAPI app
app = FastAPI(
    title="AUIChat RAG API (Version B)",
    description="Enhanced RAG API for AUIChat with hybrid retrieval, query reformulation, and multi-hop capabilities",
    version="2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model for API request
class QueryRequest(BaseModel):
    query: str

# Initialize global variables
embed_model = None
llm = None
index = None
hybrid_retriever = None
query_reformulator = None
cross_encoder_reranker = None
rag_query_engine = None

# Initialize the RAG application components
@app.on_event("startup")
async def startup():
    global llm, embed_model, index, hybrid_retriever, query_reformulator, cross_encoder_reranker, rag_query_engine
    
    try:
        # Start timer to measure initialization time
        start_time = time.time()
        
        # 1. Load the embedding model
        logger.info("Loading embedding model...")
        embed_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        
        # 2. Load the LLM
        logger.info("Loading LLM...")
        llm_model_name = os.environ.get("LLM_MODEL_NAME", "google/gemma-2b")
        llm = HuggingFaceLLM(
            model_name=llm_model_name,
            tokenizer_name=llm_model_name,
            context_window=2048,
            max_new_tokens=512,
            model_kwargs={"temperature": 0.3},  # Lower temperature for better factuality
            generate_kwargs={"do_sample": True}
        )
        
        # Update settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # 3. Load query reformulator
        logger.info("Initializing query reformulator...")
        query_reformulator = QueryReformulator()
        
        # 4. Set up cross-encoder reranker
        logger.info("Initializing cross-encoder reranker...")
        cross_encoder_reranker = CrossEncoderReranker(top_n=5)
        
        # 5. Connect to Qdrant collection or load from preprocessed nodes
        logger.info("Setting up vector store...")
        try:
            # Try connecting to Qdrant service
            from qdrant_client import QdrantClient
            
            # Check for Qdrant environment variables
            qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant-service:6333")
            qdrant_api_key = os.environ.get("QDRANT_API_KEY")
            collection_name = os.environ.get("QDRANT_COLLECTION", "AUIChatVectorCol-384")
            
            logger.info(f"Connecting to Qdrant at {qdrant_url}, collection: {collection_name}")
            
            # Initialize client with or without API key
            if qdrant_api_key:
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                client = QdrantClient(url=qdrant_url)
            
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                raise ValueError(f"Qdrant collection '{collection_name}' not found. Available collections: {collection_names}")
            
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name
            )
            
            logger.info(f"Successfully connected to Qdrant")
            
        except Exception as qdrant_error:
            logger.error(f"Error connecting to Qdrant: {qdrant_error}")
            logger.info("Attempting to load from preprocessed nodes...")
            
            try:
                # Load from the preprocessed nodes pickle file
                nodes_file = os.environ.get(
                    "PREPROCESSED_NODES",
                    "/app/preprocessed_nodes.pkl"
                )
                
                logger.info(f"Loading nodes from {nodes_file}")
                with open(nodes_file, 'rb') as f:
                    nodes = pickle.load(f)
                
                logger.info(f"Loaded {len(nodes)} nodes from {nodes_file}")
                
                # Create index from nodes directly
                index = VectorStoreIndex(nodes)
                vector_store = index.vector_store
                
                logger.info("Successfully created index from nodes")
            
            except Exception as node_error:
                logger.error(f"Could not load from preprocessed nodes: {node_error}")
                logger.error("Both Qdrant connection and node loading failed.")
                raise
        
        # Create index if not already created
        if index is None:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        
        # 6. Set up hybrid retrievers (vector + BM25)
        logger.info("Setting up hybrid retrieval...")
        
        # Create vector retriever
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=7  # Get more for hybrid approach
        )
        
        # Create BM25 retriever from the index's nodes
        all_nodes = list(index.docstore.docs.values())
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=7
        )
        
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=10,  # Will be reranked later
            weights=[0.7, 0.3],  # Balance between semantic and keyword
        )
        
        # 7. Create multi-hop query engine
        logger.info("Setting up multi-hop query engine...")
        rag_query_engine = MultiHopQueryEngine(
            retriever=hybrid_retriever,
            llm=llm,
            similarity_top_k=5
        )
        
        # Measure and log initialization time
        duration = time.time() - start_time
        logger.info(f"RAG application initialization completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during RAG application initialization: {e}", exc_info=True)
        raise

# Improved RAG prompt with better instructions
IMPROVED_PROMPT_TEMPLATE = """
Answer the following question about Al Akhawayn University (AUI) in Morocco using ONLY the context information provided below. 
Your answer should be factual, concise, and directly address the question.

If the provided context doesn't contain the information needed to answer the question fully, state clearly what information is missing 
rather than making up details. Use phrases like "The provided information doesn't specify..." or "The context doesn't mention...".

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
"""

@app.post("/predict")
async def predict(request: QueryRequest):
    """
    Process a RAG query using the enhanced Version B system
    """
    query = request.query.strip()
    if not query:
        return {"error": "Empty query", "response": "Please provide a question."}
    
    try:
        logger.info(f"Processing query: {query}")
        
        if not all([query_reformulator, hybrid_retriever, cross_encoder_reranker, rag_query_engine]):
            logger.error("RAG components not fully initialized")
            return {
                "response": random.choice(FALLBACK_RESPONSES),
                "sources": []
            }
        
        # Step 1: Query reformulation
        reformulated_query = query_reformulator.reformulate(query)
        logger.info(f"Reformulated query: {reformulated_query}")
        
        # Step 2: Multi-hop RAG query process
        response, retrieved_nodes = rag_query_engine.query(reformulated_query)
        
        # Step 3: Further reranking of the retrieved nodes
        reranked_nodes = cross_encoder_reranker.postprocess_nodes(
            retrieved_nodes,
            reformulated_query
        )
        
        # Format sources
        sources = []
        for node in reranked_nodes:
            if hasattr(node, "metadata") and node.metadata:
                source = {
                    "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                    "score": float(node.score) if hasattr(node, "score") else 0.0
                }
                
                # Add source file information if available
                if "file_name" in node.metadata:
                    source["file_name"] = node.metadata["file_name"]
                elif "source" in node.metadata:
                    source["file_name"] = node.metadata["source"]
                
                sources.append(source)
        
        # Format the final response
        response_text = response.response
        
        # Format "Step X:" to be on a new line after a period
        response_text = re.sub(r'\.\s*(Step\s+\d+:)', r'.\n\1', response_text)
        
        # Format "**Question X:" to "* Question X:"
        response_text = re.sub(r'\*\*(Question\s+\d+:)', r'* \1', response_text)
        
        logger.info(f"Returning response with {len(sources)} sources")
        return {
            "response": response_text,
            "sources": sources,
            "reformulated_query": reformulated_query,
            "original_query": query
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return {
            "response": random.choice(FALLBACK_RESPONSES),
            "sources": [],
            "original_query": query
        }

@app.post("/api/chat")
async def api_chat(request: Request):
    """API-prefixed endpoint for chat requests"""
    try:
        # Parse the request body
        body = await request.json()
        
        # Extract query from different possible formats
        query = None
        if "messages" in body and isinstance(body["messages"], list):
            # Extract the last user message
            for msg in reversed(body["messages"]):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
        elif "query" in body:
            query = body["query"]
        elif "prompt" in body:
            query = body["prompt"]
            
        if not query:
            return {"error": "No query found in request", "response": "Please provide a question."}
            
        # Reuse the predict endpoint logic
        result = await predict(QueryRequest(query=query))
        return result
        
    except Exception as e:
        logger.error(f"Error in /api/chat: {e}", exc_info=True)
        return {
            "response": random.choice(FALLBACK_RESPONSES),
            "sources": []
        }

@app.get("/health")
def health_check():
    """Health check endpoint for the RAG API"""
    components_status = {
        "llm": llm is not None,
        "embed_model": embed_model is not None,
        "index": index is not None,
        "hybrid_retriever": hybrid_retriever is not None,
        "query_reformulator": query_reformulator is not None,
        "cross_encoder_reranker": cross_encoder_reranker is not None,
        "rag_query_engine": rag_query_engine is not None
    }
    
    all_initialized = all(components_status.values())
    
    return {
        "status": "ok" if all_initialized else "initializing",
        "version": "2.0",
        "model": "Enhanced RAG (Version B)",
        "components": components_status
    }

@app.get("/")
@app.get("/api/health")
def root():
    """Root endpoint redirects to health check"""
    return health_check()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF

# Create Dockerfile
cat > "${TMP_DIR}/Dockerfile" << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up HuggingFace cache environment variables to be used during build and runtime
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache
RUN mkdir -p /app/huggingface_cache && chmod -R 777 /app/huggingface_cache

# Copy requirements file first
COPY requirements.txt .

# Install Python dependencies without hash checking
# Use multiple approaches for more resilient installation
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pip-tools retry && \
    python -c "import os; os.system('pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir --no-deps -r requirements.txt && pip install --no-cache-dir -r requirements.txt')" && \
    # Verify installation
    python -c "import llama_index; print(f'Successfully installed llama_index {llama_index.__version__}')"

# Copy the preload_models.py script
COPY preload_models.py .

# Run the script to download models into the cache layer of the image
ARG LLM_MODEL_NAME_FOR_PREBAKE_ARG=google/gemma-2b
ENV LLM_MODEL_NAME_FOR_PREBAKE=${LLM_MODEL_NAME_FOR_PREBAKE_ARG}
RUN python preload_models.py

# Copy preprocessed nodes and application code
COPY preprocessed_nodes.pkl .
COPY app.py .

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
EOF

# Create model preloading script
cat > "${TMP_DIR}/preload_models.py" << 'EOF'
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preload_models")

# Cache directory is set by ENV in Dockerfile
cache_dir = os.environ.get("HF_HOME", "/app/huggingface_cache")
logger.info(f"Using HuggingFace cache directory: {cache_dir}")

# 1. Embedding model
embed_model_name = os.environ.get("EMBEDDING_MODEL_NAME_FOR_PREBAKE", "sentence-transformers/all-MiniLM-L6-v2")
try:
    logger.info(f"Downloading embedding model: {embed_model_name}")
    SentenceTransformer(embed_model_name, cache_folder=cache_dir)
    logger.info(f"Embedding model {embed_model_name} downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download embedding model {embed_model_name}: {e}", exc_info=True)
    raise

# 2. LLM model
llm_model_name = os.environ.get("LLM_MODEL_NAME_FOR_PREBAKE")
if not llm_model_name:
    logger.error("LLM_MODEL_NAME_FOR_PREBAKE environment variable not set for preloading.")
    raise ValueError("LLM model name for pre-baking not provided.")

try:
    logger.info(f"Downloading LLM tokenizer: {llm_model_name}")
    AutoTokenizer.from_pretrained(llm_model_name, cache_dir=cache_dir)
    logger.info(f"Downloading LLM model: {llm_model_name}")
    AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir=cache_dir)
    logger.info(f"LLM {llm_model_name} downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download LLM {llm_model_name}: {e}", exc_info=True)
    # Continue even if LLM loading fails, as we have other models to load

# 3. T5 model for query reformulation
t5_model_name = "google/flan-t5-small"
try:
    logger.info(f"Downloading T5 tokenizer: {t5_model_name}")
    T5Tokenizer.from_pretrained(t5_model_name, cache_dir=cache_dir)
    logger.info(f"Downloading T5 model: {t5_model_name}")
    T5ForConditionalGeneration.from_pretrained(t5_model_name, cache_dir=cache_dir)
    logger.info(f"T5 model {t5_model_name} downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download T5 model {t5_model_name}: {e}", exc_info=True)
    # Continue even if T5 loading fails

# 4. Cross-encoder model for reranking
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
try:
    logger.info(f"Downloading cross-encoder model: {cross_encoder_model}")
    CrossEncoder(cross_encoder_model, cache_folder=cache_dir)
    logger.info(f"Cross-encoder model {cross_encoder_model} downloaded successfully.")
except Exception as e:
    logger.error(f"Failed to download cross-encoder model {cross_encoder_model}: {e}", exc_info=True)
    # Continue even if cross-encoder loading fails

logger.info("Model pre-loading script completed.")
EOF

# Create requirements.txt
cp "/home/barneh/Rag-Based-LLM_AUIChat/deployment_scripts/requirements_version_b.txt" "${TMP_DIR}/requirements.txt" #  << 'EOF'
# Core API dependencies
EOF

# Create cloudbuild.yaml
cat > "${TMP_DIR}/cloudbuild.yaml" << 'EOF'
steps:
- name: 'gcr.io/cloud-builders/docker'
  args:
  - 'build'
  - '-t'
  - '${_IMAGE_NAME}'
  - '--build-arg'
  - 'LLM_MODEL_NAME_FOR_PREBAKE_ARG=${_LLM_MODEL_NAME_FOR_PREBAKE_SUB}'
  - '--build-arg'
  - 'EMBEDDING_MODEL_NAME_FOR_PREBAKE_ARG=${_EMBEDDING_MODEL_NAME_FOR_PREBAKE_SUB}'
  - '.'
  id: 'Build Docker Image'

substitutions:
  _IMAGE_NAME: 'gcr.io/default-project/default-image'
  _LLM_MODEL_NAME_FOR_PREBAKE_SUB: 'google/gemma-2b'
  _EMBEDDING_MODEL_NAME_FOR_PREBAKE_SUB: 'sentence-transformers/all-MiniLM-L6-v2'

# Increase the build timeout to handle model downloads
timeout: '3600s'

images:
- '${_IMAGE_NAME}'
EOF

# Build and deploy to Cloud Run
echo "🔨 Building container image using cloudbuild.yaml (this may take a while due to model downloads)..."
# Select model for prebaking - use a smaller model for faster builds if needed
LLM_FOR_PREBAKE=${LLM_MODEL_NAME:-"google/gemma-2b"}
EMBEDDING_FOR_PREBAKE=${EMBEDDING_MODEL_NAME:-"sentence-transformers/all-MiniLM-L6-v2"}

if [ "$DRY_RUN" = "true" ]; then
  echo "🔍 DRY RUN: Would submit build with the following configuration:"
  echo "  - Image name: ${IMAGE_NAME}"
  echo "  - LLM model: ${LLM_FOR_PREBAKE}"
  echo "  - Embedding model: ${EMBEDDING_FOR_PREBAKE}"
  echo "  - cloudbuild.yaml content:"
  cat "${TMP_DIR}/cloudbuild.yaml"
else
  gcloud builds submit "${TMP_DIR}" \
    --config "${TMP_DIR}/cloudbuild.yaml" \
    --substitutions "_IMAGE_NAME=${IMAGE_NAME},_LLM_MODEL_NAME_FOR_PREBAKE_SUB=${LLM_FOR_PREBAKE},_EMBEDDING_MODEL_NAME_FOR_PREBAKE_SUB=${EMBEDDING_FOR_PREBAKE}"
fi

echo "🚀 Deploying to Cloud Run"
# Retrieve QDRANT_URL and QDRANT_API_KEY from environment or use defaults
QDRANT_URL_TO_USE=${QDRANT_URL:-"https://40003a30-70d7-4886-9de5-45e25681c36e.europe-west3-0.gcp.cloud.qdrant.io"}
QDRANT_API_KEY_TO_USE=${QDRANT_API_KEY:-"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uea3Q5G9lcLfqCwxzTpRKWcMh5XM0pvPB2RaeOaDPxM"}
QDRANT_COLLECTION_TO_USE=${QDRANT_COLLECTION:-"AUIChatVectorCol-384"}

if [ "$DRY_RUN" = "true" ]; then
  echo "🔍 DRY RUN: Would deploy to Cloud Run with the following configuration:"
  echo "  - Service name: ${SERVICE_NAME}"
  echo "  - Image: ${IMAGE_NAME}"
  echo "  - Region: ${REGION}"
  echo "  - Memory: 4Gi"
  echo "  - CPU: 2"
  echo "  - Timeout: 600s"
  echo "  - CPU Boost: enabled"
  echo "  - Min instances: 0"
  echo "  - Max instances: 5"
  echo "  - Authentication: public (--allow-unauthenticated)"
  echo "  - Environment variables:"
  echo "    - QDRANT_URL: ${QDRANT_URL_TO_USE}"
  echo "    - QDRANT_API_KEY: [HIDDEN]"
  echo "    - QDRANT_COLLECTION: ${QDRANT_COLLECTION_TO_USE}"
  echo "    - LLM_MODEL_NAME: ${LLM_MODEL_NAME:-google/gemma-2b}"
  echo "    - EMBEDDING_MODEL_NAME: ${EMBEDDING_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"
  
  # Simulate a service URL for dry run
  SERVICE_URL="https://${SERVICE_NAME}---${REGION}.run.app"
else
  gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --memory 4Gi \
    --cpu 2 \
    --timeout 600s \
    --cpu-boost \
    --min-instances 0 \
    --max-instances 5 \
    --allow-unauthenticated \
    --set-env-vars="QDRANT_URL=${QDRANT_URL_TO_USE}" \
    --set-env-vars="^##^QDRANT_API_KEY=${QDRANT_API_KEY_TO_USE}" \
    --set-env-vars="QDRANT_COLLECTION=${QDRANT_COLLECTION_TO_USE}" \
    --set-env-vars="LLM_MODEL_NAME=${LLM_MODEL_NAME:-google/gemma-2b}" \
    --set-env-vars="EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}" \
    --no-cpu-throttling

  # Get the service URL
  SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --format="value(status.url)")
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --platform managed \
  --region "${REGION}" \
  --format="value(status.url)")

echo "✅ Deployment successful!"
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "To use this endpoint in your code, set the environment variable:"
echo "export VERSION_B_ENDPOINT_URL=${SERVICE_URL}"
echo ""
echo "To run A/B test against your production RAG system:"
echo "python ${PROJECT_ROOT}/rag_based_llm_auichat/ML6/run_ab_testing.py --endpoint-a <PRODUCTION_ENDPOINT> --endpoint-b ${SERVICE_URL}"

# Save deployment info to project root
cat > "${PROJECT_ROOT}/version_b_info.json" << EOF
{
  "service_name": "${SERVICE_NAME}",
  "service_url": "${SERVICE_URL}",
  "project_id": "${PROJECT_ID}",
  "region": "${REGION}",
  "image_name": "${IMAGE_NAME}",
  "version": "B",
  "features": [
    "Hybrid Retrieval (vector + BM25)",
    "Query Reformulation",
    "Advanced Re-ranking with cross-encoders",
    "Multi-hop RAG architecture",
    "Improved prompting",
    "Better generation parameters"
  ]
}
EOF
echo "Deployment info saved to ${PROJECT_ROOT}/version_b_info.json"
