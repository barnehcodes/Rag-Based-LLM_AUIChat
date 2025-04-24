"""
Model loader module for AUIChat
This module handles loading the SmolLM-360M model locally
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def create_local_llm_handler():
    """
    Creates and returns a text-generation pipeline using the local SmolLM-360M model.
    This replaces the Hugging Face Inference API calls with local model inference.
    """
    model_id = "HuggingFaceTB/SmolLM-360M-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SmolLM-360M model on {device}...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    
    print("✅ SmolLM-360M model loaded successfully")
    return pipe