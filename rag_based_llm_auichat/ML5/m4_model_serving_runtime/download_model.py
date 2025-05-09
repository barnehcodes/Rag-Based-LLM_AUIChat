#!/usr/bin/env python3
"""
Script to download embedding model for local packaging
"""
import os
import argparse
from sentence_transformers import SentenceTransformer

def download_model(model_name, output_dir):
    """Download a SentenceTransformer model and save it to the specified directory"""
    print(f"Downloading model: {model_name}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download and save the model
    model = SentenceTransformer(model_name)
    model_path = os.path.join(output_dir, model_name.replace("/", "_"))
    model.save(model_path)
    
    print(f"Model saved to: {model_path}")
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download embedding model for local packaging")
    parser.add_argument("--model-name", type=str, default="BAAI/bge-small-en-v1.5",
                       help="Hugging Face model name")
    parser.add_argument("--output-dir", type=str, default="./models",
                       help="Output directory for the downloaded model")
    
    args = parser.parse_args()
    download_model(args.model_name, args.output_dir)
