#!/usr/bin/env python3
# Simple script to test the RAG functions

import pickle
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG-Test")

def query_rag_system() -> Dict[str, Any]:
    """
    Query the RAG system directly to test functionality
    """
    try:
        # Path to preprocessed nodes
        nodes_path = "../../rag_based_llm_auichat/preprocessed_nodes.pkl"
        question = "What are the counseling services available at AUI?"
        
        # Set up basic components manually
        from llama_index.core import VectorStoreIndex, StorageContext, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        # Load preprocessed nodes
        print(f"Loading nodes from {nodes_path}")
        try:
            with open(nodes_path, "rb") as f:
                nodes = pickle.load(f)
                print(f"Loaded {len(nodes)} nodes from {nodes_path}")
        except FileNotFoundError:
            # Try another location
            nodes_path = "../../preprocessed_nodes.pkl"
            with open(nodes_path, "rb") as f:
                nodes = pickle.load(f)
                print(f"Loaded {len(nodes)} nodes from {nodes_path}")
        
        # Debug nodes
        print(f"Node type: {type(nodes)}, Count: {len(nodes)}")
        if nodes:
            print(f"First node type: {type(nodes[0])}")
            print(f"First node sample text: {nodes[0].text[:100]}...")
        
        # Setup embedding model
        print("Setting up embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model
        
        # Setup LLM
        print("Setting up T5 model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", legacy=False)
        
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=512,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7},
            is_chat_model=False,
        )
        Settings.llm = llm
        print("LLM setup complete")
        
        # Create storage context and index
        print("Creating index...")
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)
        print("Index created")
        
        # Create query engine
        print("Creating query engine...")
        query_engine = index.as_query_engine(
            similarity_top_k=3,
        )
        print("Query engine created")
        
        # Make the query
        print(f"Querying with: {question}")
        start_time = time.time()
        response = query_engine.query(question)
        end_time = time.time()
        print(f"Response received in {end_time - start_time:.2f} seconds")
        print(f"Response: {str(response)}")
        
        # Extract contexts
        contexts = []
        if hasattr(response, "source_nodes"):
            print(f"Found {len(response.source_nodes)} source nodes")
            for node in response.source_nodes:
                contexts.append({
                    "text": node.node.text,
                    "score": node.score if hasattr(node, "score") else 0.0,
                    "file_name": node.node.metadata.get("file_name", "Unknown")
                })
                print(f"Context: {node.node.text[:100]}... Score: {node.score}")
        else:
            print("No source nodes found in response")
            print(f"Response attributes: {dir(response)}")
        
        return {
            "answer": str(response),
            "contexts": contexts,
            "latency": end_time - start_time,
            "success": True
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "contexts": [],
            "latency": 0,
            "success": False
        }

if __name__ == "__main__":
    print("Testing RAG function...")
    result = query_rag_system()
    print("\nFinal result:")
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer']}")
    print(f"Contexts: {len(result['contexts'])}")
    print(f"Latency: {result['latency']:.2f}s")
