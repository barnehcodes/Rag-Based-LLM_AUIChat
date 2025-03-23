# pipeline/feature_store.py
from zenml import step
from feast import FeatureStore
import os
from datetime import datetime
import pandas as pd

@step
def ingest_features(
    repo_path: str = "./feast_repo",
    start_date: str = "2021-01-01",
    end_date: str = "2021-12-31"
) -> str:
    """
    Loads the Feast repository configuration from the YAML file, applies
    the feature definitions, and materializes the offline features to the online store.
    """
    fs = FeatureStore(repo_path=repo_path)
    # Use the keyword argument 'repo_yaml' so Feast loads the configuration correctly.
    fs.apply(repo_yaml=os.path.join(repo_path, "feature_store.yaml"))
    # Optionally materialize features from the offline source to the online store.
    fs.materialize(start_date=start_date, end_date=end_date)
    print("✅ Features successfully ingested into Feast!")
    return "features_ingested"

@step
def get_features(entity_dict: dict, feature_refs: list) -> "pd.DataFrame":
    """
    Retrieves historical features from Feast's offline store based on an entity DataFrame
    and a list of feature references.
    
    Args:
        entity_dict: A dictionary of entities, e.g. {"driver_id": [1001, 1002], "event_timestamp": ["2021-04-12T10:00:00", "2021-04-12T10:00:00"]}
        feature_refs: A list of feature references in the format "<feature_view>:<feature_name>"
    
    Returns:
        A pandas DataFrame containing the retrieved features.
    """
    repo_path = "./feast_repo"
    fs = FeatureStore(repo_path=repo_path)
    # Convert the input entity dictionary to a DataFrame.
    entity_df = pd.DataFrame(entity_dict)
    # Retrieve historical features (point-in-time join).
    feature_data = fs.get_historical_features(
        entity_df=entity_df,
        features=feature_refs
    ).to_df()
    print("✅ Features retrieved from Feast:")
    print(feature_data.head())
    return feature_data
