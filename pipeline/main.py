# main.py
from zenml import pipeline
from Data_preprocessing import preprocess_data
from index_storage import create_and_store_index
from validation import validate_qdrant_storage
from query_engine import query_qdrant  # Now returns response text
from feature_store import ingest_features
import gradio as gr

@pipeline
def auichat_data_pipeline(query: str):
    """Full pipeline for AUIChat that runs preprocessing, indexing, feature ingestion, validation, and querying."""
    nodes_file = preprocess_data()
    create_and_store_index(nodes_file)
    ingest_features()
    validation_result = validate_qdrant_storage()
    
    if validation_result:
        results = query_qdrant(query)
        if isinstance(results, list) and results:
            return results[0]["text"]
        return "✅ Qdrant returned but no text found."
    else:
        return "🚨 Qdrant storage validation failed."

# Gradio chatbot wrapper
def chat_with_aui_chatbot(user_input):
    print(f"User Input: {user_input}")
    response = auichat_data_pipeline(query=user_input)
    return response

if __name__ == "__main__":
    gr.ChatInterface(chat_with_aui_chatbot,
                     title="AUIChat 🤖",
                     description="Ask me anything about AUI Policies & Services!",
                     theme="default").launch()
