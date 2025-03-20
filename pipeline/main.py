from zenml import pipeline
from Data_preprocessing import preprocess_data
from index_storage import create_and_store_index
from validation import validate_qdrant_storage
from query_engine import query_qdrant  # Query step now decorated with @step

@pipeline
def auichat_data_pipeline(query: str):
    """Full pipeline for AUIChat that runs preprocessing, indexing, validation, and querying."""
    nodes_file = preprocess_data()  # This step returns the pickle file path
    create_and_store_index(nodes_file)  # Index is created and stored in Qdrant
    validation_result = validate_qdrant_storage()
    
    if validation_result:
        results = query_qdrant(query)
        return results
    else:
        return "Storage validation failed"

if __name__ == "__main__":
    user_query = "What are the requirements for the PiP program?"
    auichat_data_pipeline(query=user_query)
