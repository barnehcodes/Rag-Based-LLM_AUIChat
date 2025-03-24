# engines/gradio_ui.py
from zenml import step
import gradio as gr
from engines.query_engine import create_query_engine

@step
def launch_gradio_interface():
    """Launches the Gradio-based AUIChat interface."""

    def chat(query):
        response = create_query_engine(query)
        return str(response)

    iface = gr.Interface(
        fn=chat,
        inputs=gr.Textbox(label="Ask AUIChat"),
        outputs=gr.Textbox(label="Response"),
        title="AUIChat - RAG Based LLM Chatbot",
        description="Ask questions based on the indexed university data"
    )

    print("🚀 Launching AUIChat Gradio UI...")
    iface.launch()
