# feature_store.py
from zenml import step
from feast import FeatureStore
import os

@step
def ingest_features(
    repo_path: str = "./Feast",
    start_date: str = "2021-01-01",
    end_date: str = "2021-12-31"
) -> str:
    """
    Ingests features into Feast and materializes them into the online store.

    Args:
        repo_path: Path to your Feast repository.
        start_date: Start date for feature materialization.
        end_date: End date for feature materialization.

    Returns:
        A status message indicating the result of the ingestion.
    """
    # Initialize the Feast feature store with your repo path
    fs = FeatureStore(repo_path=repo_path)
    
    # Apply your feature definitions (this can load a YAML file or Python-defined features)
    fs.apply(os.path.join(repo_path, "feature_store.yaml"))
    
    # Materialize features from your offline store to the online store
    fs.materialize(start_date=start_date, end_date=end_date)
    
    print("✅ Features successfully ingested into Feast!")
    return "features_ingested"
