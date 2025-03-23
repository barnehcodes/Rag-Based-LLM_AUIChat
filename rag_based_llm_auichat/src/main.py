# pipeline/main.py
from zenml import pipeline
from data.Data_preprocessing import preprocess_data
from data.index_storage import create_and_store_index
from features.feature_store import ingest_features, get_features  # Import both Feast steps
from valisation.validation import validate_qdrant_storage
from engines.query_engine import query_qdrant  # Query step for Qdrant

@pipeline
def auichat_data_pipeline(query: str, entity_dict: dict, feature_refs: list):
    """
    Full pipeline for AUIChat that runs preprocessing, indexing, 
    feature ingestion/retrieval, storage validation, and querying.
    """
    # Preprocess data and create an index in Qdrant.
    nodes_file = preprocess_data()  # Returns the pickle file path
    create_and_store_index(nodes_file)  # Creates and stores index in Qdrant

    # Ingest feature definitions and optionally materialize features into Feast.
    feast_status = ingest_features()  
    
    # Retrieve offline features from Feast based on provided entity data and feature references.
    features_artifact = get_features(entity_dict=entity_dict, feature_refs=feature_refs)
    # Unwrap the artifact using .value (if supported by your ZenML version)
    features_df = features_artifact.value
    
    # Validate that Qdrant storage is working.
    validation_result = validate_qdrant_storage()
    
    if validation_result:
        results = query_qdrant(query)
        print("Retrieved Feast features:")
        print(features_df.head())
        return results
    else:
        return "Storage validation failed"

if __name__ == "__main__":
    user_query = "What are the requirements for the PiP program?"
    
    # Example entity dictionary for offline feature retrieval.
    entity_dict = {
        "driver_id": [1001, 1002],
        "event_timestamp": ["2021-04-12T10:00:00", "2021-04-12T10:00:00"]
    }
    
    # Feature references in the format "<feature_view>:<feature_name>".
    feature_refs = [
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate",
        "driver_hourly_stats:avg_daily_trips"
    ]
    
    # Run the pipeline.
    auichat_data_pipeline(query=user_query, entity_dict=entity_dict, feature_refs=feature_refs)
