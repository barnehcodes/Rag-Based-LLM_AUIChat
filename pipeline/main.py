# main.py
from Data_preprocessing import process_documents
from index_storage import create_and_store_index
from validation import validate_qdrant_storage
from query_engine import query_qdrant, create_query_engine

def main():
    # Process documents (text cleaning & chunking)
    nodes = process_documents()
    
    # Create and store index in Qdrant
    index = create_and_store_index(nodes)
    
    # Validate Qdrant storage
    print("\n--- Validating Qdrant Storage ---")
    storage_valid = validate_qdrant_storage()
    
    if storage_valid:
        print("\n--- Running Example Direct Query ---")
        query_qdrant("What are the requirements for the PiP program?")
        
        print("\n--- Running Example RAG Query Engine ---")
        create_query_engine("What are the requirements for the PiP program?")
    else:
        print("🚨 Storage validation failed - please check your Qdrant configuration")

if __name__ == "__main__":
    main()
