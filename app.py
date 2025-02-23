import os
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr

# Hugging Face Inference API settings
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from your PDFs
pip_requirements_text = extract_text_from_pdf("PiP 24-25 Program Requirements.pdf")
counseling_faq_text = extract_text_from_pdf("Counseling Services FAQ Spring 2024.pdf")

# Combine all texts into a single knowledge base
knowledge_base = counseling_faq_text + "\n" + pip_requirements_text

# Step 2: Set up retrieval
# Split the knowledge base into chunks (e.g., paragraphs or sentences)
knowledge_chunks = knowledge_base.split("\n")

# Encode the chunks using a sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(knowledge_chunks)

# Create a FAISS index
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])  # L2 distance
index.add(chunk_embeddings)

# Step 3: Define the retrieval function
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [knowledge_chunks[i] for i in indices[0]]
    return relevant_chunks

# Step 4: Define the response generation function using the Inference API
def generate_response(query):
    # Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)
    context = "\n".join(relevant_chunks)

    # Truncate the context if it's too long
    max_context_length = 256  # Adjust as needed
    if len(context) > max_context_length:
        context = context[:max_context_length]

    # Combine query and context for Mistral 7B
    input_text = f"Context: {context}\n\nQuestion: {query}"

    # Send the input to the Hugging Face Inference API
    payload = {"inputs": input_text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Step 5: Create Gradio UI
def chatbot_interface(query):
    return generate_response(query)

# Suggested queries for quick testing
suggested_queries = [
    "Who can use AUI Counseling Services?",
    "What are the eligibility requirements for the PiP program?",
    "What types of counseling services are available at AUI?",
    "How do I set up an appointment with a counselor?",
]

# Gradio interface
interface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
    outputs=gr.Textbox(lines=4, label="Chatbot Response"),
    title="University Chatbot",
    description="Ask questions about university services and programs.",
    examples=suggested_queries,
)

# Launch the interface
interface.launch()