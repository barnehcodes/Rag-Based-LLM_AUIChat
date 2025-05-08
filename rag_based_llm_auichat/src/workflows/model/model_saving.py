from zenml import step
from zenml.logger import get_logger
import mlflow
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import datetime
import tempfile

# Use the correct HuggingFace model ID instead of a local path
MODEL_ID = "HuggingFaceTB/SmolLM-360M-Instruct"
MLFLOW_MODEL_NAME = "auichat-smollm-360m"

logger = get_logger(__name__)

@step(enable_cache=False)
def save_model_for_deployment() -> str:
    """
    Loads the SmolLM model and tokenizer from Hugging Face Hub and saves them using MLflow.

    Returns:
        The MLflow URI of the saved model artifact.
    """
    logger.info(f"⏳ Loading model and tokenizer from {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer and model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        logger.info("✅ Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading model/tokenizer from {MODEL_ID}: {e}")
        raise

    logger.info("⏳ Saving model with MLflow...")
    task = "text-generation"

    # Define explicit pip requirements instead of auto-discovery
    pip_requirements = [
        f"torch=={torch.__version__}",
        f"transformers=={transformers.__version__}",
        "sentencepiece",
        "protobuf"
    ]

    # Create a unique model path with timestamp to avoid conflicts
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_model_path = f"{MLFLOW_MODEL_NAME}_{timestamp}"
    
    # Create a temporary directory for saving the model
    temp_dir = tempfile.mkdtemp(prefix="mlflow_model_")
    model_path = os.path.join(temp_dir, unique_model_path)
    
    logger.info(f"Using unique model path: {model_path}")

    # Use standard MLflow saving
    with mlflow.start_run() as run:
        try:
            mlflow.transformers.save_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                path=model_path,  # Use the unique path
                task=task,
                pip_requirements=pip_requirements
            )
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/{unique_model_path}"  # Update the URI to match the path
            logger.info(f"✅ Model saved successfully in MLflow run: {run_id}")
            logger.info(f"   Model URI: {model_uri}")
            return model_uri
        except Exception as e:
            logger.error(f"❌ Error saving model with MLflow: {e}")
            mlflow.end_run(status="FAILED")
            raise
