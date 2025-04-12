from zenml import step
import json
from src.workflows.config import qdrant_client, COLLECTION_NAME, embed_model, vector_store
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
import mlflow
import mlflow.sklearn  # or whichever flavor you're tracking

@step
def query_qdrant(query_text: str, limit: int = 5):
    """Queries Qdrant and prints the top-k relevant documents."""
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
    
    results = []
    for i, result in enumerate(search_results):
        print(f"\nResult {i+1} - Score: {result.score}")
        print(f"Document ID: {result.id}")
        
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
            results.append({"score": result.score, "file": file_name, "text": text[:500]})
        else:
            print("No text found in payload")
            print(f"Available payload keys: {list(result.payload.keys())}")
        
        print("-" * 50)
    
    return results

@step
def create_query_engine(query_text: str):
    """Creates a query engine using the Hugging Face Inference API for RAG queries."""
    try:
        # Initialize the LLM model
        llm = HuggingFaceInferenceAPI(
            model_name="mistralai/Mistral-7B-Instruct-v0.3", 
            token="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"
        )
        
        # Configure llama-index Settings
        from llama_index.core import Settings
        Settings.embed_model = embed_model
        
        # Setup vector store
        from src.workflows.config import qdrant_client, COLLECTION_NAME
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
        
        # Create an index with the vector store
        index = VectorStoreIndex.from_vector_store(temp_vector_store)
        
        # Create a query engine using the inference API LLM
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,  # Retrieve top 3 most similar chunks
            streaming=False
        )
        
        # Run the query
        print(f"📝 Processing query: {query_text}")
        response = query_engine.query(query_text)
        
        # Log metrics for ZenML/MLflow tracking
        try:
            source_nodes = getattr(response, 'source_nodes', [])
            retrieved_chunks = len(source_nodes) if source_nodes else 0
            
            mlflow.log_metric("response_length", len(str(response)))
            mlflow.log_metric("retrieved_chunks", retrieved_chunks)
            
            # Save query + response for review
            with open("query_response.txt", "w") as f:
                f.write(f"Query: {query_text}\n\nResponse:\n{response}")
                
                if source_nodes:
                    f.write("\n\nSources:\n")
                    for i, node in enumerate(source_nodes):
                        f.write(f"\nSource {i+1}:\n{node.get_text()[:500]}...\n")
                        
            mlflow.log_artifact("query_response.txt")
        except Exception as log_err:
            print(f"Warning: Could not log metrics: {str(log_err)}")
        
        print(f"✅ Response generated successfully")
        return response
        
    except Exception as e:
        print(f"❌ Error in query engine: {str(e)}")
        # Return a helpful error message that can be displayed to the user
        return f"I encountered an error while processing your query. Please try again or contact support.\nError: {str(e)}"