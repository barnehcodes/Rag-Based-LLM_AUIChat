# gradio_app.py
import gradio as gr
from engines.query_engine import create_query_engine

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

if __name__ == "__main__":
    iface.launch()
