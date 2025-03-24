from zenml import pipeline
from data.Data_preprocessing import preprocess_data
from data.index_storage import create_and_store_index
from valisation.validation import validate_qdrant_storage
from engines.query_engine import query_qdrant
from workflows.evaluation import evaluate_response
from engines.gradio_ui import launch_gradio_interface

@pipeline
def auichat_data_pipeline(query: str):
    """
    Full pipeline for AUIChat that runs preprocessing, indexing,
    validation, querying, evaluation, and launches Gradio UI.
    """
    nodes_file = preprocess_data()
    create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage()

    if validation_result:
        query_result = query_qdrant(query)
        evaluate_response(query=query, llm_response=query_result)
        launch_gradio_interface()
    else:
        print("❌ Qdrant storage validation failed.")

if __name__ == "__main__":
    user_query = "What are the requirements for the PiP program?"
    auichat_data_pipeline(query=user_query)
