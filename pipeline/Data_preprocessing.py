# preprocessing.py
import re
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def clean_text(text):
    """Cleans and normalizes text before embedding."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9,.!?;:\'\"()\[\]\s]', '', text)
    return text

def process_documents(directory="resources/"):
    print(f"Loading documents from {directory}")
    documents = SimpleDirectoryReader(directory).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Apply smart chunking
    text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")
    
    # Apply cleaning to chunks and set metadata
    for node in nodes:
        node.text = clean_text(node.text)
        # Include full text in metadata for easy retrieval
        node.metadata = {
            "file_name": node.metadata.get("file_name", "Unknown"),
            "page_label": node.metadata.get("page_label", "Unknown"),
            "text": node.text,  # Store text directly in metadata
            "chunk_id": str(node.id_)  # Add ID for reference
        }
    
    # Verify chunk sizes
    chunk_sizes = [len(node.text.split()) for node in nodes]
    print(f"Min Chunk Size: {min(chunk_sizes)} words")
    print(f"Max Chunk Size: {max(chunk_sizes)} words")
    print(f"Average Chunk Size: {sum(chunk_sizes)/len(chunk_sizes):.2f} words")
    
    return nodes
