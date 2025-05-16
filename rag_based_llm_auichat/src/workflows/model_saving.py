from zenml import step, get_step_context # Added get_step_context
from zenml.client import Client # Added Client
from zenml.logger import get_logger
import mlflow
import transformers
# Remove direct AutoModelForCausalLM, AutoTokenizer imports if LocalLLM handles them
# from transformers import AutoModelForCausalLM, AutoTokenizer 
import os
import torch
import datetime
import tempfile
import pandas as pd # Added for predict method input handling

import mlflow.pyfunc
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
# Ensure these can be initialized correctly in the Seldon environment
# Remove QDRANT_PORT as it's not used for cloud Qdrant and not in config.py
from rag_based_llm_auichat.src.workflows.config.config import qdrant_client, embed_model, vector_store, COLLECTION_NAME, QDRANT_HOST 
from rag_based_llm_auichat.src.engines.local_models.local_llm import LocalLLM 

# Configuration for the RAG model
# MODEL_ID is used by LocalLLM, ensure it's defined or LocalLLM knows how to get it.
# If LocalLLM doesn't use this MODEL_ID directly, remove it from here.
# MODEL_ID = "HuggingFaceTB/SmolLM-360M-Instruct" # This should be handled by LocalLLM

SIMILARITY_TOP_K = 8  # Increased from 5 to retrieve more potentially relevant documents
CUSTOM_QA_TEMPLATE_STR = """SYSTEM: You are a helpful AI assistant for Al Akhawayn University. Your task is to answer the question using ONLY the information in the context below. 
If you cannot find the answer in the context, say "I don't have enough information about this in my knowledge base."
Respond with factual information only. Do not make up information. Format your answer as plain text without including metadata or debugging information.

CONTEXT:
{context_str}

USER: {query_str}

ASSISTANT:"""

logger = get_logger(__name__)

class RAGChatbotModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logger.info("RAGChatbotModel: Initializing context...")
        
        # 1. Initialize LLM (via LocalLLM)
        # LocalLLM should handle loading the actual SmolLM model and tokenizer.
        # It might need MODEL_ID or other configurations passed or set as environment variables.
        self.llm = LocalLLM() 
        logger.info("RAGChatbotModel: LLM initialized.")

        # 2. Initialize Embedding Model (imported as embed_model from src.workflows.config.config)
        # This assumes embed_model from config.py is correctly initialized upon import.
        # If config.py needs to be re-run or its initializers called, that logic would go here.
        self.embed_model = embed_model 
        if not self.embed_model:
            # Attempt to initialize if not already done - this is a fallback
            logger.warning("RAGChatbotModel: embed_model not initialized, attempting to re-initialize from config.")
            # This part is tricky and depends on how config.py is structured.
            # For simplicity, we assume direct import works or it's already initialized.
            # from src.workflows.config.config import initialize_embedding_model # Hypothetical
            # self.embed_model = initialize_embedding_model()
            if not self.embed_model: # Check again
                 raise ValueError("Embedding model could not be initialized. Check src.workflows.config.config and its dependencies.")
        logger.info("RAGChatbotModel: Embedding model configured.")

        # 3. Set LlamaIndex global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        logger.info("RAGChatbotModel: LlamaIndex Settings configured.")

        # 4. Initialize Vector Store and Index
        # This assumes vector_store from config.py is correctly initialized.
        # Qdrant client and vector_store might need re-initialization if they don't persist.
        self.vector_store = vector_store
        if not self.vector_store:
            logger.warning("RAGChatbotModel: vector_store not initialized, attempting to re-initialize from config.")
            # from src.workflows.config.config import initialize_qdrant_vector_store # Hypothetical
            # self.vector_store = initialize_qdrant_vector_store()
            if not self.vector_store: # Check again
                raise ValueError("Vector store could not be initialized. Check src.workflows.config.config, Qdrant connection, and env vars (QDRANT_HOST, QDRANT_PORT).")

        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        logger.info("RAGChatbotModel: VectorStoreIndex initialized.")

        # 5. Define QA Prompt Template
        self.qa_template = PromptTemplate(CUSTOM_QA_TEMPLATE_STR)
        logger.info("RAGChatbotModel: QA Prompt Template defined.")

        # 6. Create Query Engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            text_qa_template=self.qa_template,
            streaming=False 
        )
        logger.info(f"RAGChatbotModel: Query engine created with similarity_top_k={SIMILARITY_TOP_K}.")
        logger.info("RAGChatbotModel: Context loaded successfully.")

    def predict(self, context, model_input):
        logger.info(f"RAGChatbotModel: Received input for prediction: {type(model_input)} - {model_input}")
        query = None
        if isinstance(model_input, pd.DataFrame):
            # Standard MLflow pyfunc behavior: DataFrame input
            # Assuming query is in the first column, first row
            if not model_input.empty and not model_input.iloc[0].empty:
                 query = str(model_input.iloc[0,0])
            else:
                logger.error("RAGChatbotModel: Received empty DataFrame.")
                return {"error": "Received empty DataFrame."}
        elif isinstance(model_input, dict): # Common for direct API calls to Seldon
            # Try to extract query from common patterns
            if "query" in model_input:
                query = model_input["query"]
            # Seldon's /predict endpoint often wraps input in a 'data' field
            elif "data" in model_input and isinstance(model_input["data"], dict):
                if "ndarray" in model_input["data"] and len(model_input["data"]["ndarray"]) > 0:
                    query = model_input["data"]["ndarray"][0]
                elif "tensor" in model_input["data"] and "values" in model_input["data"]["tensor"] and len(model_input["data"]["tensor"]["values"]) > 0: # TF serving format
                    query = model_input["data"]["tensor"]["values"][0]
            # Seldon V2 protocol uses "inputs"
            elif "inputs" in model_input and isinstance(model_input["inputs"], list) and len(model_input["inputs"]) > 0:
                input_payload = model_input["inputs"][0]
                if "data" in input_payload and len(input_payload["data"]) > 0:
                     query = input_payload["data"][0] # Assuming string data

        if not isinstance(query, str) or not query.strip():
            logger.error(f"RAGChatbotModel: Query not found or invalid in model_input: {model_input}")
            return {"error": f"Query not found or invalid in input. Expecting DataFrame or dict with 'query' or Seldon specific structure. Received: {model_input}"}

        logger.info(f"RAGChatbotModel: Processing query: '{query}'")
        try:
            # Special debug logging for admission-related queries
            is_admission_query = False
            if "admission" in query.lower() or "apply" in query.lower() or "application" in query.lower():
                is_admission_query = True
                logger.info("ADMISSION QUERY DETECTED: Enhanced debugging enabled")
            
            response = self.query_engine.query(query)
            
            # Enhanced logging for contexts
            source_nodes_summary = []
            if hasattr(response, 'source_nodes'):
                logger.info(f"Retrieved {len(response.source_nodes)} context nodes")
                
                for i, node in enumerate(response.source_nodes):
                    node_text = node.get_text()
                    node_score = node.get_score()
                    node_id = node.node_id
                    
                    # Create a source node summary
                    source_summary = {
                        "text_snippet": node_text[:100] + "...",
                        "score": node_score,
                        "id": node_id
                    }
                    source_nodes_summary.append(source_summary)
                    
                    # For admission queries, log more detailed context information
                    if is_admission_query:
                        logger.info(f"CONTEXT NODE {i+1}/{len(response.source_nodes)}:")
                        logger.info(f"  Score: {node_score}")
                        # Log more of the text for admission queries
                        logger.info(f"  Text: {node_text[:500]}...")
                        
                        # Check if node metadata contains useful info
                        if hasattr(node, 'metadata') and node.metadata:
                            logger.info(f"  Metadata: {node.metadata}")
            
            # Clean up the response to remove any debug information
            response_text = str(response)
            
            # Clean up response by removing any metadata or file path information
            # This will catch patterns like "page_label:", "file_path:", etc.
            import re
            
            # Patterns to clean up
            patterns = [
                r'page_label:.*?\n',
                r'file_path:.*?\n',
                r'In the context above.*?\n',
                r'We have provided an updated answer:.*?\n'
            ]
            
            # Apply each cleanup pattern
            for pattern in patterns:
                response_text = re.sub(pattern, '', response_text)
                
            # Remove references to the question being asked in the response
            question_patterns = [
                rf'{re.escape(query)}.*?\n',
                r'What counseling services are available at AUI\?.*?\n',
                r'How do I apply for undergraduate admission\?.*?\n'
            ]
            
            for pattern in question_patterns:
                response_text = re.sub(pattern, '', response_text)
                
            # Remove any trailing or leading whitespace
            response_text = response_text.strip()
            
            logger.info(f"RAGChatbotModel: Cleaned query response: {response_text}")
            return {"response": response_text, "source_nodes": source_nodes_summary}
        except Exception as e:
            logger.error(f"RAGChatbotModel: Error during query processing: {e}", exc_info=True)
            return {"error": str(e)}

@step(enable_cache=False)
def save_model_for_deployment(model_name: str = "auichat-rag-model") -> str:
    """
    Saves the RAG chatbot as an MLflow pyfunc model.
    
    Args:
        model_name: Name to use for the saved MLflow pyfunc model.
        
    Returns:
        The MLflow URI of the saved model artifact.
    """
    logger.info(f"⏳ Preparing to save RAG model with name: {model_name}")

    # Attempt to get MLflow tracking URI from ZenML active stack first
    # This is useful for logging and understanding the environment.
    active_stack_mlflow_uri = None
    try:
        active_stack = Client().active_stack
        experiment_tracker = active_stack.experiment_tracker
        if (experiment_tracker and 
            hasattr(experiment_tracker.config, 'tracking_uri') and 
            experiment_tracker.config.tracking_uri):
            active_stack_mlflow_uri = experiment_tracker.config.tracking_uri
            logger.info(f"MLflow tracking URI from ZenML active stack (will be overridden if forcing specific path): {active_stack_mlflow_uri}")
        else:
            logger.warning(
                "MLflow tracking URI not found in ZenML active stack's experiment tracker. "
                "Will use explicitly defined path."
            )
    except Exception as e:
        logger.warning(f"Could not get MLflow tracking URI from ZenML active stack: {e}. Will use explicitly defined path.")

    # Explicitly set the MLflow tracking URI to the notebooks/mlruns directory.
    # This will be the definitive URI for this step, overriding any from the stack for this step's context.
    notebooks_mlruns_path = "file:///home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/notebooks/mlruns/"
    try:
        mlflow.set_tracking_uri(notebooks_mlruns_path)
        logger.info(f"Successfully FORCED MLflow tracking URI to: {notebooks_mlruns_path}")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to set MLflow tracking URI to '{notebooks_mlruns_path}': {e}. This step cannot proceed.")
        raise

    # Define an experiment name for saved models
    experiment_name = "AUIChat-Registered-Models"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"Successfully set MLflow experiment to: '{experiment_name}'")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to set or access MLflow experiment '{experiment_name}': {e}. This is often due to a corrupted meta.yaml file for an existing experiment MLflow is trying to load. Please check your mlruns directory. This step cannot proceed without a valid experiment setup.")
        raise # Re-raise the exception to stop the pipeline here, as start_run will likely fail.

    # Pip requirements for the RAG model environment
    pip_requirements = [
        f"torch=={torch.__version__}",
        f"transformers=={transformers.__version__}",
        "sentencepiece",
        "protobuf",
        f"mlflow=={mlflow.__version__}",
        "llama-index>=0.10.0,<0.11.0", # Specify your project's version range
        "qdrant-client>=1.7.0,<1.9.0", # Specify your project's version range
        "sentence-transformers>=2.2.0,<3.0.0", # Specify version for embed_model
        "pandas" # Added for predict method
        # Add other specific dependencies from your project, e.g., python-dotenv if config.py uses it
        # "python-dotenv"
    ]
    
    # Path for MLflow to log the model artifact. This is relative to the run's artifact root.
    pyfunc_artifact_path = model_name 
    
    logger.info(f"Using pyfunc artifact path for MLflow logging: {pyfunc_artifact_path}")

    with mlflow.start_run(run_name=model_name) as run:
        try:
            # Log the RAGChatbotModel instance
            # The `code_path` argument bundles specified local code with the model.
            # This is crucial if `src` is not an installed package in the Seldon environment.
            # Assuming this script is run from a context where "./src" is valid (e.g., project root)
            # or ZenML handles pathing.
            # For robustness, ensure paths in code_path are resolvable.
            # If your ZenML pipeline executes from the root of "Rag-Based-LLM_AUIChat",
            # then "rag_based_llm_auichat/src" would be the path to your "src" module.
            # Let's assume the current working directory or PYTHONPATH allows `import src...`
            # If `src` is part of a package `rag_based_llm_auichat` that is installed,
            # code_path might not be strictly needed for `src` itself but for uninstalled local modules.
            # Given the project structure, it's likely `src` is a top-level importable if PYTHONPATH is set up.
            # We will rely on PYTHONPATH being correctly set by ZenML or the execution environment.
            # If issues arise, `code_path=["./src"]` or `code_path=["./rag_based_llm_auichat/src"]`
            # (relative to execution dir) might be needed.
            mlflow.pyfunc.log_model(
                artifact_path=pyfunc_artifact_path,
                python_model=RAGChatbotModel(),
                pip_requirements=pip_requirements,
                # Example: code_path=["./rag_based_llm_auichat/src"] # if needed
            )
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/{pyfunc_artifact_path}"
            
            logger.info(f"✅ RAG Pyfunc Model saved successfully in MLflow run: {run_id}")
            logger.info(f"   Model URI for Seldon: {model_uri}")
            return model_uri
        except Exception as e:
            logger.error(f"❌ Error saving RAG Pyfunc model with MLflow: {e}", exc_info=True)
            mlflow.end_run(status="FAILED")
            raise

# Remove the old MODEL_ID constant if it's no longer used directly in this file
# MODEL_ID = "HuggingFaceTB/SmolLM-360M-Instruct" # Commented out as LocalLLM should handle its model ID

# The rest of the file (imports like AutoModelForCausalLM, AutoTokenizer)
# might be cleaned up if they are no longer directly used.
# For now, keeping them doesn't harm if they are part of transformers import.
