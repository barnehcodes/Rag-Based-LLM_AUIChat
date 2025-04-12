#!/usr/bin/env python
"""
Debug script to identify errors in the RAG system
"""
import os
import sys
from pathlib import Path

# Add the src directory to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

print("Step 1: Testing imports...")
try:
    from src.workflows.config import load_environment, qdrant_client, COLLECTION_NAME, embed_model
    print("✅ Config imports successful")
except Exception as e:
    print(f"❌ Config import error: {str(e)}")
    sys.exit(1)

print("\nStep 2: Testing environment setup...")
try:
    load_environment()
    print("✅ Environment loaded successfully")
except Exception as e:
    print(f"❌ Environment loading error: {str(e)}")
    sys.exit(1)

print("\nStep 3: Testing Qdrant connection...")
try:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"✅ Connected to Qdrant collection '{COLLECTION_NAME}' with {collection_info.points_count} vectors")
except Exception as e:
    print(f"❌ Qdrant connection error: {str(e)}")
    sys.exit(1)

print("\nStep 4: Testing query engine import...")
try:
    from src.engines.query_engine import create_query_engine
    print("✅ Query engine import successful")
except Exception as e:
    print(f"❌ Query engine import error: {str(e)}")
    sys.exit(1)

print("\nStep 5: Testing query execution...")
try:
    test_query = "What are the requirements for the PiP program?"
    print(f"Executing test query: '{test_query}'")
    
    # Create a direct RAG query without using ZenML step
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
    query_engine = index.as_query_engine(llm=llm)
    
    # Execute query
    result = query_engine.query(test_query)
    
    print(f"✅ Query executed successfully. Response length: {len(str(result))}")
    print(f"Response preview: {str(result)[:100]}...")
except Exception as e:
    print(f"❌ Query execution error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed successfully! Your RAG system is working.")