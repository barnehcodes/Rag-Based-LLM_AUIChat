# query_engine.py
import json
from config import qdrant_client, COLLECTION_NAME, embed_model, vector_store
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os

def query_qdrant(query_text, limit=5):
    # Generate embedding for the query
    query_vector = embed_model.get_text_embedding(query_text)
    
    # Run similarity search in Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
    
    print(f"Found {len(search_results)} results for query: '{query_text}'")
    
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1} - Score: {result.score}")
        print(f"Document ID: {result.id}")
        
        # Try different payload locations for text
        text = None
        if 'text' in result.payload:
            text = result.payload.get('text')
            file_name = result.payload.get('metadata', {}).get('file_name', result.payload.get('file_name', 'Unknown'))
        elif 'metadata' in result.payload and 'text' in result.payload['metadata']:
            text = result.payload['metadata']['text']
            file_name = result.payload['metadata'].get('file_name', 'Unknown')
        elif '_node_content' in result.payload:
            node_content = json.loads(result.payload.get('_node_content', '{}'))
            text = node_content.get("text", "No text found")
            file_name = node_content.get('metadata', {}).get('file_name', 'Unknown')
        elif 'content' in result.payload:
            content = result.payload['content']
            if isinstance(content, str):
                try:
                    content_data = json.loads(content)
                    text = content_data.get('text', content)
                except:
                    text = content
            else:
                text = str(content)
            file_name = result.payload.get('file_name', 'Unknown')
        
        if text:
            print(f"File: {file_name}")
            print(f"Text: {text[:500]}...")
        else:
            print("No text found in payload")
            print(f"Available payload keys: {list(result.payload.keys())}")
        
        print("-" * 50)

def create_query_engine(query_text):
    """Create a query engine for proper RAG queries using Hugging Face Inference API."""
    # Initialize LLM to use the Inference API; ensure your API token is set in HUGGINGFACEHUB_API_TOKEN
    #api_key = os.getenv("HUGGING_FACE_API_KEY")
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.3", 
        token=#API TOKEN
    )
    from llama_index.core import Settings
    
    # Initialize settings for the query
    Settings.embed_model = embed_model
    
    # Create vector store using the same parameters as before
    from config import qdrant_client, COLLECTION_NAME  # re-import if needed
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    temp_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        text_key="text",
        metadata_key="metadata",
        content_key="content",
        embed_dim=768,
        stores_text=True
    )
    
    # Create an empty index with the vector store
    index = VectorStoreIndex.from_vector_store(temp_vector_store)
    
    # Create a query engine using the inference API LLM
    query_engine = index.as_query_engine(llm=llm)
    
    # Execute query
    response = query_engine.query(query_text)
    
    print(f"\nRAG Response for query: '{query_text}'")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    return response
