"""
API Handler for the RAG-Based LLM AUIChat System
"""
import traceback
import random
from typing import List, Dict, Any
import sys
import os

# Add project root to sys.path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import from other modules - use absolute imports
from rag_based_llm_auichat.src.engines.local_models.local_llm import LocalLLM
from rag_based_llm_auichat.src.workflows.config import index, embed_model

# Create FastAPI app
app = FastAPI(title="RAG-Based LLM AUIChat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The conversation history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The assistant's response")
    sources: List[Dict[str, str]] = Field(default=[], description="Sources used for the response")

# Define fallback responses
FALLBACK_RESPONSES = [
    "I apologize, but I'm having trouble retrieving that information at the moment. Could you try asking in a different way?",
    "I don't have enough information to answer that question properly. Could you provide more details or ask something else?",
    "I'm sorry, but I couldn't find reliable information to answer your question. Please try a different question or contact the university directly.",
    "That's a good question, but I'm not able to provide accurate information on that right now. Could we try a different topic?",
    "I'm still learning about Al Akhawayn University. I don't have enough context to answer that question properly yet."
]

@app.get("/api/health") # Changed route from "/" to "/api/health"
def health_check_api():
    """Health check endpoint for API path"""
    # You might want to add checks here, e.g., if the index is loaded
    status = {
        "status": "ok", 
        "message": "AUIChat API is running",
        "index_loaded": index is not None,
        # Add qdrant check if possible/needed
    }
    return status

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Process a chat request and generate a response using the RAG pipeline
    """
    # Extract the last user message
    try:
        last_message = next((m for m in reversed(request.messages) if m.role == "user"), None)
        if not last_message:
            raise ValueError("No user message found in the conversation history")
        
        query = last_message.content
        
        # Check if the pre-built index is available from config
        if index is None:
            print("⚠️ Pre-built index not available, using fallback response")
            return ChatResponse(
                response=random.choice(FALLBACK_RESPONSES),
                sources=[]
            )
        
        # Create a query engine using the pre-built index and local LLM
        print(f"Creating query engine with pre-built index...")
        query_engine = index.as_query_engine(
            llm=LocalLLM(),
            similarity_top_k=3,
            structured_answer_filtering=True
        )
        
        # Execute the query
        response = query_engine.query(query)
        
        # Extract sources if available
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if hasattr(node, 'metadata'):
                    source = {
                        "file_name": node.metadata.get("file_name", "Unknown"),
                        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text
                    }
                    sources.append(source)
        
        return ChatResponse(
            response=str(response),
            sources=sources
        )
    
    except Exception as e:
        # If any error occurs, try up to 3 times then use fallback responses
        print(f"⚠️ Query attempt failed: {str(e)}")
        traceback.print_exc()
        
        return ChatResponse(
            response=random.choice(FALLBACK_RESPONSES),
            sources=[]
        )

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000)