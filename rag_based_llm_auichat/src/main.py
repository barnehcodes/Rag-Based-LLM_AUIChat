from zenml import pipeline, get_step_context
from zenml.client import Client
from zenml.logger import get_logger # Import ZenML logger
import sys
import os

# Add the directory containing m5_frontend_client, m7_deployment, src, etc. to sys.path
# This is the parent directory of the 'src' directory where main.py is located.
path_to_add = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_to_add)

logger = get_logger(__name__) # Initialize logger for main.py

# --- Existing Imports ---
from ML5.m5_frontend_client.ui_launcher import launch_ui_components
from ML5.m7_deployment.firebase_ui_deployment import deploy_ui_to_firebase
# Import the Seldon components for the local deployment pipeline
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService, 
)
from zenml.integrations.seldon.steps import seldon_model_deployer_step

# Import our step functions
from src.data.Data_preprocessing import preprocess_data
from src.data.index_storage import create_and_store_index
from src.valisation.validation import validate_qdrant_storage
from src.workflows.model_saving import save_model_for_deployment
from src.workflows.model_training import placeholder_model_trainer

# Import our new step functions
from src.workflows.mlflow_utils import launch_mlflow_dashboard_step
from src.workflows.data_validation import validate_processed_data_step
from src.workflows.cloud_testing import test_cloud_run_endpoint_step
from src.workflows.ui_build import build_ui_for_firebase_step
# from src.workflows.custom_cloud_run_deployment import deploy_improved_rag_app_step

# Define constants for deployment
SELDON_DEPLOYMENT_NAME = "auichat-smollm-deployment-local"
MLFLOW_MODEL_NAME = "auichat-smollm-360m" # Used by save_model_for_deployment
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "AUIChatVectoreCol-384")
PROJECT_ID = os.environ.get("PROJECT_ID", "deft-waters-458118-a3")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", PROJECT_ID)

# Activate the GCP stack if running with cloud resources
def activate_gcp_stack():
    """Activates the GCP stack for cloud deployments"""
    try:
        client = Client()
        # Check if gcp-stack exists
        stacks = client.list_stacks()
        if "gcp-stack" in [s.name for s in stacks]:
            client.activate_stack("gcp-stack")
            logger.info("✅ Activated GCP stack for cloud resources")
            return True
        else:
            logger.warning("⚠️ GCP stack not found. Using default stack.")
            return False
    except Exception as e:
        logger.warning(f"⚠️ Error activating GCP stack: {e}")
        logger.warning("⚠️ Continuing with default stack")
        return False

# ------------------------------------------------------------------------------------
# Local Deployment Pipeline (via Seldon and Kubernetes)
# ------------------------------------------------------------------------------------
@pipeline(name="LOCAL_AUICHAT_DEPLOYMENT_PIPELINE", enable_cache=False)
def auichat_local_deployment_pipeline():
    """
    Local deployment pipeline for AUIChat using Seldon Core on Kubernetes.
    Covers the full development lifecycle from data processing to UI launch.
    """
    logger.info("🚀 Starting Local AUIChat Deployment Pipeline (Seldon/Kubernetes)...")

    # --- BIG STEP 1: DATA_ACQUISITION_VALIDATION_AND_PREPARATION ---
    logger.info("--- BIG STEP 1: DATA_ACQUISITION_VALIDATION_AND_PREPARATION ---")
    
    # Preprocess the data (load, clean, chunk documents, and save nodes to disk)
    processed_nodes_file = preprocess_data()
    
    # Validate the processed data
    validation_status = validate_processed_data_step(nodes_file_path=processed_nodes_file)
    
    # Index and store the data
    index_creation_status = create_and_store_index(
        nodes_file=processed_nodes_file, 
        after=[validation_status]
    )
    
    # Validate Qdrant storage
    qdrant_validation_status = validate_qdrant_storage(after=[index_creation_status])

    # --- BIG STEP 2: MODEL_TRAINING_AND_EVALUATION ---
    logger.info("--- BIG STEP 2: MODEL_TRAINING_AND_EVALUATION ---")
    
    # Placeholder for model training step
    training_status = placeholder_model_trainer(after=[qdrant_validation_status])
    
    # Save the model using MLflow
    model_uri = save_model_for_deployment()
    
    # Launch the MLflow dashboard
    mlflow_ui_status = launch_mlflow_dashboard_step(after=[model_uri])

    # --- BIG STEP 3: ML_PRODUCTIONIZATION ---
    logger.info("--- BIG STEP 3: ML_PRODUCTIONIZATION ---")
    
    # Extract model_uri as a string from the artifact
    # In ZenML, we often need to convert step output artifacts to their actual values
    model_uri_str = model_uri if isinstance(model_uri, str) else str(model_uri)
    logger.info(f"Model URI for Seldon deployment: {model_uri_str}")
    
    # Model serving/runtime using ZenML and Seldon
    seldon_deployment_config = SeldonDeploymentConfig(
        name=SELDON_DEPLOYMENT_NAME,
        model_uri=model_uri_str,
        replicas=1,
        implementation="MLFLOW_SERVER",
        parameters=[
            {
                "name": "model_uri",
                "value": model_uri_str,
                "type": "STRING"
            }
        ],
        resources={
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "1000m", "memory": "2Gi"}
        },
        secret_name="seldon-init-container-secret"  # Default secret name used by Seldon
    )
    
    # Deploy to local Seldon
    seldon_service = seldon_model_deployer_step(
        model=model_uri,                   # Pass the model artifact
        service_config=seldon_deployment_config,  # Use service_config instead of config
        after=[mlflow_ui_status]
    )
    
    # Front-end client using ui_launcher
    ui_status = launch_ui_components(after=[seldon_service])
    
    logger.info("✅ Local AUIChat Deployment Pipeline completed.")

# ------------------------------------------------------------------------------------
# Cloud Deployment Pipeline (via Cloud Run and Firebase)
# ------------------------------------------------------------------------------------
@pipeline(name="CLOUD_AUICHAT_DEPLOYMENT_PIPELINE", enable_cache=False)
def auichat_cloud_deployment_pipeline():
    """
    Cloud deployment pipeline for AUIChat using Cloud Run for the backend
    and Firebase for hosting the UI.
    """
    logger.info("🚀 Starting Cloud AUIChat Deployment Pipeline (Cloud Run/Firebase)...")
    activate_gcp_stack() # Activate GCP stack for cloud deployments

    # --- STEP 1: DATA_PREPARATION ---
    logger.info("--- BIG STEP: DATA_PREPARATION_FOR_CLOUD_ENVIRONMENT ---")
    
    # Preprocess the data
    processed_nodes_file_cloud = preprocess_data()
    
    # Validate the processed data
    validation_status_cloud = validate_processed_data_step(
        nodes_file_path=processed_nodes_file_cloud
    )
    
    # Index and store the data
    index_creation_status_cloud = create_and_store_index(
        nodes_file=processed_nodes_file_cloud,
        after=[validation_status_cloud]
    )
    
    # Validate Qdrant storage
    qdrant_validation_status_cloud = validate_qdrant_storage(
        after=[index_creation_status_cloud]
    )

    # --- STEP 2: DEPLOY_MODEL_TO_CLOUD_RUN ---
    logger.info("--- BIG STEP: DEPLOY_RAG_BACKEND_TO_CLOUD_RUN ---")
    
    # Deploy the model to a Cloud Run endpoint using improved_rag_app_qdrant.py
    cloud_run_deployment_info = deploy_improved_rag_app_step(
        project_id=PROJECT_ID,
        region=GCP_REGION,
        service_name="auichat-rag-prod",
        after=[qdrant_validation_status_cloud]
    )

    # --- STEP 3: TEST_CLOUD_RUN_ENDPOINT ---
    logger.info("--- BIG STEP: TEST_CLOUD_RUN_ENDPOINT ---")
    
    # Test the endpoint via a POST request
    endpoint_test_status = test_cloud_run_endpoint_step(
        cloud_run_deployment_info=cloud_run_deployment_info
    )

    # --- STEP 4: BUILD_UI_AND_HOST_UI_IN_FIREBASE ---
    logger.info("--- BIG STEP: BUILD_AND_DEPLOY_UI_TO_FIREBASE ---")
    
    # Build the UI if the dist directory doesn't already exist
    ui_dist_path = build_ui_for_firebase_step(after=[endpoint_test_status])
    
    # Host the UI in Firebase
    firebase_deployment_info = deploy_ui_to_firebase(
        project_id=FIREBASE_PROJECT_ID,
        backend_url=cloud_run_deployment_info.get("service_url"),
        after=[ui_dist_path]
    )
    
    logger.info("✅ Cloud AUIChat Deployment Pipeline completed.")

# --- Main execution block ---
if __name__ == "__main__":
    pipeline_choice = os.environ.get("AUICHAT_PIPELINE_CHOICE", "local").lower()
    
    if len(sys.argv) > 1:
        pipeline_choice = sys.argv[1].lower()

    logger.info(f"Selected pipeline: '{pipeline_choice}'")

    if pipeline_choice == "local":
        logger.info("🚀 Running Local Deployment Pipeline (Seldon/Kubernetes)...")
        auichat_local_deployment_pipeline()
    elif pipeline_choice == "cloud":
        logger.info("🚀 Running Cloud Deployment Pipeline (Cloud Run/Firebase)...")
        auichat_cloud_deployment_pipeline()
    else:
        logger.error(f"Unknown pipeline choice: '{pipeline_choice}'. Please choose 'local' or 'cloud'.")
        logger.info("Example: python src/main.py local")
        logger.info("Or set environment variable: export AUICHAT_PIPELINE_CHOICE=cloud")
