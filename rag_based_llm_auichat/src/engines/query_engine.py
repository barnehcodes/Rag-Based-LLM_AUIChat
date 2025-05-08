from zenml import step
import json
# Ensure config imports the correct, updated values
from src.workflows.config.config import qdrant_client, COLLECTION_NAME, embed_model, vector_store, EMBED_DIM 
from llama_index.core import VectorStoreIndex, Settings
import os
import mlflow
import mlflow.sklearn  # or whichever flavor you're tracking

# Import the local model handler instead of the HuggingFace API
from src.engines.local_models.local_llm import LocalLLM

# Note: query_qdrant might be less relevant if queries only come via the UI/API
# but can be kept for debugging or direct pipeline interaction if needed.
@step
def query_qdrant(query_text: str, limit: int = 5):
    """Queries Qdrant using the configured embed_model and COLLECTION_NAME."""
    if not qdrant_client or not embed_model:
        print("❌ Qdrant client or embedding model not initialized. Skipping query.")
        return []
        
    # Generate embedding for the query using the configured model (BGE)
    query_vector = embed_model.get_text_embedding(query_text)
    
    # Run similarity search in Qdrant using the configured collection
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME, # Use updated collection name from config
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
    
    print(f"Found {len(search_results)} results for query: '{query_text}'")
    
    results = []
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1} - Score: {result.score}")
        print(f"Document ID: {result.id}")
        
        text = None
        # Adjusted payload access logic based on potential structures
        payload = result.payload if hasattr(result, 'payload') else {}
        
        if 'text' in payload:
            text = payload.get('text')
            file_name = payload.get('metadata', {}).get('file_name', payload.get('file_name', 'Unknown'))
        elif 'metadata' in payload and 'text' in payload['metadata']:
            text = payload['metadata']['text']
            file_name = payload['metadata'].get('file_name', 'Unknown')
        elif '_node_content' in payload:
            try:
                node_content = json.loads(payload.get('_node_content', '{}'))
                text = node_content.get("text", "No text found")
                file_name = node_content.get('metadata', {}).get('file_name', 'Unknown')
            except json.JSONDecodeError:
                text = "Error decoding _node_content"
                file_name = "Unknown"
        elif 'content' in payload:
            content = payload['content']
            if isinstance(content, str):
                try:
                    content_data = json.loads(content)
                    text = content_data.get('text', content)
                except json.JSONDecodeError:
                    text = content
            else:
                text = str(content)
            file_name = payload.get('file_name', 'Unknown')
        
        if text:
            print(f"File: {file_name}")
            print(f"Text: {text[:500]}...")
            results.append({"score": result.score, "file": file_name, "text": text[:500]})
        else:
            print("No text found in payload")
            print(f"Available payload keys: {list(payload.keys())}")
        
        print("-" * 50)
    
    return results

# This step might be redundant if the query engine is only used by the API,
# but can be kept if direct pipeline querying is needed.
@step
def create_query_engine(query_text: str):
    """Creates a query engine using the local SmolLM-360M model and configured vector store/embedding model."""
    try:
        if not vector_store or not embed_model:
             raise ConnectionError("Vector store or embedding model not initialized. Check config.py and Qdrant connection.")
             
        # Initialize the local LLM model
        llm = LocalLLM()
        
        # Configure llama-index Settings globally for this step
        # Ensure the correct embedding model (BGE) is used
        Settings.embed_model = embed_model
        Settings.llm = llm # Set the local LLM globally for LlamaIndex
        
        # Create an index directly from the existing vector store (loaded in config.py)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Create a query engine using the local LLM
        query_engine = index.as_query_engine(
            # llm=llm, # LLM is now set globally via Settings
            similarity_top_k=3,  # Retrieve top 3 most similar chunks
            streaming=False
        )
        
        # Run the query
        print(f"Executing query with local LLM: '{query_text}'")
        response = query_engine.query(query_text)
        print("Query finished.")
        
        # Log query and response (optional)
        # with mlflow.start_run():
        #     mlflow.log_param("query", query_text)
        #     mlflow.log_text(str(response), "response.txt")
            
        print(f"Response: {str(response)}")
        return str(response)
        
    except Exception as e:
        print(f"❌ Error creating/using query engine: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"