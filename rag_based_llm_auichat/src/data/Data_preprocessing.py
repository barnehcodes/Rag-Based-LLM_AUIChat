from zenml import step
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
import re
import json
import pickle

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9,.!?;:\'\"()\[\]\s]', '', text)
    return text

@step
def preprocess_data(directory: str = "rag_based_llm_auichat/data/raw") -> str:
    """Loads, cleans, and chunks documents, then saves the nodes to disk."""
    print(f"📂 Loading documents from {directory}...")
    documents = SimpleDirectoryReader(directory).load_data()
    
    text_splitter = SentenceSplitter(chunk_size=450, chunk_overlap=50)
    nodes = text_splitter.get_nodes_from_documents(documents)

    # Apply cleaning
    for node in nodes:
        node.text = clean_text(node.text)
    
    # Save nodes to a file using pickle
    output_file = "preprocessed_nodes.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(nodes, f)
    
    print(f"✅ Preprocessed {len(nodes)} document chunks and saved to {output_file}")
    return output_file  # Return the file path (a JSON-serializable string)
