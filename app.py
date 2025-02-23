from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr
import torch

import os
from huggingface_hub import login
from dotenv import load_dotenv

# Log in to Hugging Face
# Load environment variables from .env file
load_dotenv()
Access_Token = os.getenv("Access_Token")
login(token=Access_Token)

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Debug: Check if files exist
print("Files in directory:", os.listdir("/home/user/app/"))

# Extract text from your PDFs
pip_requirements_text = extract_text_from_pdf("/home/user/app/PiP 24-25 Program Requirements.pdf")
counseling_faq_text = extract_text_from_pdf("/home/user/app/Counseling Services FAQ Spring 2024.pdf")

# # Extract text from your PDFs
# pip_requirements_text = extract_text_from_pdf("PiP 24-25 Program Requirements.pdf")
# counseling_faq_text = extract_text_from_pdf("Counseling Services FAQ Spring 2024.pdf")

# Combine all texts into a single knowledge base
knowledge_base = counseling_faq_text + "\n" + pip_requirements_text

# Step 2: Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
)

# Load Mistral 7B with 8-bit quantization
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,  # Pass the quantization config
)

# Step 3: Set up retrieval
# Split the knowledge base into chunks (e.g., paragraphs or sentences)
knowledge_chunks = knowledge_base.split("\n")

# Encode the chunks using a sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(knowledge_chunks)

# Create a FAISS index
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])  # L2 distance
index.add(chunk_embeddings)

# Step 4: Define the retrieval function
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [knowledge_chunks[i] for i in indices[0]]
    return relevant_chunks

# Step 5: Define the response generation function
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
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    outputs = model.generate(**inputs, max_length=128)  # Generate shorter responses
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Step 6: Create Gradio UI
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