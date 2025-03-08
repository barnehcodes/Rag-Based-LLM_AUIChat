from zenml import pipeline
from pipeline.Data_preprocessing import preprocess_data
from pipeline.index_storage import create_and_store_index
from pipeline.validation import validate_qdrant_storage
# from pipeline.query_engine import query_qdrant

@pipeline
def auichat_data_pipeline():
    """Full pipeline for AUIChat that runs preprocessing, indexing, validation, and querying."""
    nodes_file = preprocess_data()
    create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage()
    
    # if validation_result:
    #     results = query_qdrant(query)
    #     return results
    # else:
    #     return "Storage validation failed"

if __name__ == "__main__":
    # user_query = "What are the requirements for the PiP program?"
    # auichat_data_pipeline(query=user_query)
    auichat_data_pipeline()