from zenml import pipeline, get_step_context
from zenml.client import Client
import sys
import os

# Add the directory containing m5_frontend_client, m7_deployment, src, etc. to sys.path
# This is the parent directory of the 'src' directory where main.py is located.
path_to_add = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path_to_add)

from m5_frontend_client.ui_launcher import launch_ui_components
from m7_deployment.vertex_deployment import deploy_model_to_vertex
from m7_deployment.cloudrun_deployment_step import deploy_cloudrun_rag_service
# Import the new UI deployment steps
from m7_deployment.firebase_ui_deployment import deploy_ui_to_firebase
from m7_deployment.cloudrun_ui_deployment import deploy_ui_to_cloudrun

# Fix imports with proper absolute paths
# Import data modules with proper project-relative paths
from data.Data_preprocessing import preprocess_data
from data.index_storage import create_and_store_index
from valisation.validation import validate_qdrant_storage
from workflows.model_saving import save_model_for_deployment
from workflows.model_training import placeholder_model_trainer
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService,
)
from zenml.integrations.seldon.steps import seldon_model_deployer_step
# Import our development test step
# from ..tests.development_test_step import run_development_tests # Commented out

# Define constants for deployment
SELDON_DEPLOYMENT_NAME = "auichat-smollm-deployment"
MLFLOW_MODEL_NAME = "auichat-smollm-360m"
QDRANT_COLLECTION_NAME = "AUIChatVectoreCol-384" # Added for vector search

# Activate the GCP stack if running with cloud resources
def activate_gcp_stack():
    """Activates the GCP stack for cloud deployments"""
    try:
        client = Client()
        # Check if gcp-stack exists
        stacks = client.list_stacks()
        if "gcp-stack" in [s.name for s in stacks]:
            client.activate_stack("gcp-stack")
            print("✅ Activated GCP stack for cloud resources")
            return True
        else:
            print("⚠️ GCP stack not found. Using default stack.")
            return False
    except Exception as e:
        print(f"⚠️ Error activating GCP stack: {e}")
        print("⚠️ Continuing with default stack")
        return False

# Define the Seldon deployment pipeline
@pipeline(enable_cache=False)
def auichat_seldon_deployment_pipeline():
    """
    Deployment pipeline for AUIChat that runs preprocessing, indexing,
    validation, saves the model via MLflow, deploys it to Seldon,
    and launches the UI components.
    """
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Save the model using MLflow, returns the URI string
    model_uri = save_model_for_deployment(after=[validation_result])
    
    # Try to deploy with Seldon
    try:
        # Define the Seldon deployment configuration
        deployment_config = SeldonDeploymentConfig(
            model_name=SELDON_DEPLOYMENT_NAME,
            implementation="TRANSFORMER",
            parameters=[],
            resources={
                "requests": {"cpu": "100m", "memory": "512Mi"},
                "limits": {"cpu": "1000m", "memory": "1Gi"}
            },
            replicas=1,
            secret_name="",
        )

        # Deploy the model to Seldon
        deployment_service = seldon_model_deployer_step(
            model=model_uri,
            service_config=deployment_config,
            after=[model_uri]
        )
        print("✅ Model deployed successfully with Seldon Core!")
        
        # Launch UI components after deployment
        launch_ui_components(after=[deployment_service])
    except Exception as e:
        print(f"❌ Seldon deployment failed: {str(e)}")
        print("⚠️ Continuing with UI launch without Seldon deployment.")
        
        # Launch UI components without deployment
        launch_ui_components(after=[model_uri])

# Define the Vertex AI deployment pipeline
@pipeline(enable_cache=False)
def auichat_vertex_deployment_pipeline():
    """
    Deployment pipeline for AUIChat that uses Google Cloud Vertex AI
    for model deployment instead of Seldon Core.
    """
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Save the model using MLflow, returns the URI string
    model_uri = save_model_for_deployment(after=[validation_result])
    
    # Deploy with Vertex AI
    try:
        # Extract the model URI string from StepArtifact
        if hasattr(model_uri, 'read'):
            # This is the proper way to access the content of a StepArtifact
            model_uri_str = model_uri.read()
            print(f"📦 Extracted model URI for deployment: {model_uri_str}")
        elif hasattr(model_uri, 'uri'):
            # Alternative way to access the URI if read() isn't available
            model_uri_str = model_uri.uri
            print(f"📦 Extracted model URI from uri attribute: {model_uri_str}")
        elif isinstance(model_uri, dict) and 'uri' in model_uri:
            # If model_uri is already a dictionary
            model_uri_str = model_uri['uri']
            print(f"📦 Extracted model URI from dictionary: {model_uri_str}")
        elif isinstance(model_uri, str):
            # If model_uri is already a string
            model_uri_str = model_uri
            print(f"📦 Using model URI directly as string: {model_uri_str}")
        else:
            # Convert to string as last resort
            model_uri_str = str(model_uri)
            print(f"⚠️ Using string representation of model_uri: {model_uri_str}")
        
        # Deploy the model to a Vertex AI endpoint with minimal resources
        deployment_info = deploy_model_to_vertex(
            model_uri=model_uri_str,  # Pass the extracted string URI
            machine_type="n1-standard-2",  # Low-cost machine type
            min_replicas=1,
            max_replicas=1,
            after=[model_uri]
        )
        
        # Check if deployment was successful
        if hasattr(deployment_info, 'read'):
            # If deployment_info is a ZenML StepArtifact, try to read its content
            try:
                deployment_data = deployment_info.read()
                print(f"✅ Model deployed successfully on Vertex AI!")
                print(f"   Deployment information available in ZenML dashboard.")
                # Pass the deployment_info to the UI launcher
                launch_ui_components(after=[deployment_info])
            except Exception as e:
                print(f"⚠️ Could not read deployment info: {str(e)}")
                print("⚠️ Continuing with UI launch without model deployment.")
                launch_ui_components(after=[model_uri])
        elif isinstance(deployment_info, dict):
            # If deployment_info is a dictionary
            endpoint_url = deployment_info.get('endpoint_url', 'URL not available')
            if endpoint_url != 'deployment_failed':
                print(f"✅ Model deployed successfully on Vertex AI!")
                print(f"   Endpoint URL: {endpoint_url}")
                # Launch UI components after deployment
                launch_ui_components(after=[deployment_info])
            else:
                error_msg = deployment_info.get('error', 'Unknown error')
                print(f"❌ Vertex AI deployment failed: {error_msg}")
                print("⚠️ Continuing with UI launch without model deployment.")
                # Launch UI components without deployment
                launch_ui_components(after=[model_uri])
        else:
            print(f"⚠️ Unexpected deployment_info type: {type(deployment_info)}.")
            print("⚠️ Continuing with UI launch without model deployment.")
            # Launch UI components without deployment
            launch_ui_components(after=[model_uri])
            
    except Exception as e:
        print(f"❌ Vertex AI deployment failed: {str(e)}")
        print("⚠️ Continuing with UI launch without model deployment.")
        
        # Launch UI components without deployment
        launch_ui_components(after=[model_uri])

# Pipeline that skips the deployment step entirely
@pipeline(enable_cache=False)
def auichat_no_deploy_pipeline():
    """
    Simplified pipeline for AUIChat that runs preprocessing, indexing,
    validation, saves the model via MLflow, and launches the UI components
    without attempting deployment.
    """
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Save the model using MLflow, returns the URI string
    model_uri = save_model_for_deployment(after=[validation_result])
    
    # Launch UI components after model saving
    launch_ui_components(after=[model_uri])

# Add our Cloud Run RAG deployment pipeline with vector search
@pipeline(enable_cache=False)
def auichat_cloudrun_rag_pipeline():
    """
    Deployment pipeline for AUIChat that uses Google Cloud Run
    for deploying a RAG service with Qdrant vector similarity search.
    """
    # Run the preprocessing and indexing steps
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Deploy the RAG service to Cloud Run with vector search
    try:
        # Deploy to Cloud Run using our new step
        cloudrun_info = deploy_cloudrun_rag_service(
            preprocessed_nodes_path=nodes_file,
            collection_name=QDRANT_COLLECTION_NAME,
            after=[validation_result]
        )
        
        print(f"✅ RAG service deployed to Cloud Run!")
        if isinstance(cloudrun_info, dict) and 'service_url' in cloudrun_info:
            print(f"   Service URL: {cloudrun_info['service_url']}")
        
        # Launch UI components after deployment
        launch_ui_components(after=[cloudrun_info])
        
    except Exception as e:
        print(f"❌ Cloud Run deployment failed: {str(e)}")
        print("⚠️ Continuing with UI launch without deployment.")
        
        # Launch UI components without deployment
        launch_ui_components(after=[nodes_file])

# Add our development testing pipeline
@pipeline(enable_cache=False)
def auichat_development_test_pipeline():
    """
    Development pipeline for AUIChat that runs extensive tests to ensure
    the system is properly set up before launching the UI.
    
    This pipeline checks:
    1. If the Qdrant collection is properly populated
    2. If the embedding model is working correctly for vector similarity
    3. If the RAG endpoint is responding to queries as expected
    """
    # Run the development tests
    test_results = run_development_tests(
        collection_name=QDRANT_COLLECTION_NAME,
        min_vectors=500
    )
    
    # When working with ZenML StepArtifacts, we cannot directly access dictionary keys
    # So we just print a message and continue with launching the UI
    print("\n✓ Development tests executed!")
    print("Check the logs above for detailed test results.")
    print("The UI will now be launched for development purposes.")
    
    # Always launch the UI (even if tests failed, for debugging purposes)
    launch_ui_components(after=[test_results])

# Add Firebase UI deployment pipeline
@pipeline(enable_cache=False)
def auichat_firebase_ui_pipeline():
    """
    Pipeline for deploying the AUIChat UI to Firebase Hosting.
    Firebase Hosting is ideal for static web applications like React.
    """
    # First run the development tests to ensure everything is working
    # test_results = run_development_tests(
    #     collection_name=QDRANT_COLLECTION_NAME,
    #     min_vectors=500
    # )
    
    # Deploy UI to Firebase Hosting
    firebase_info = deploy_ui_to_firebase(
        project_id=os.environ.get("PROJECT_ID", None)
        # after=[test_results] # Removed dependency on test_results
    )
    
    print("\n✅ UI deployment pipeline completed!")
    print("Check the deployment info above for the hosting URL.")

# Add Cloud Run UI deployment pipeline
@pipeline(enable_cache=False)
def auichat_cloudrun_ui_pipeline():
    """
    Pipeline for deploying the AUIChat UI to Google Cloud Run.
    Cloud Run allows for more complex setups and server-side logic if needed.
    """
    # First run the development tests to ensure everything is working
    test_results = run_development_tests(
        collection_name=QDRANT_COLLECTION_NAME,
        min_vectors=500
    )
    
    # Deploy UI to Cloud Run
    cloudrun_info = deploy_ui_to_cloudrun(
        project_id=os.environ.get("PROJECT_ID", None),
        region=os.environ.get("REGION", "europe-west3"),
        after=[test_results]
    )
    
    print("\n✅ UI deployment pipeline completed!")
    print("Check the deployment info above for the service URL.")

# Add a full deployment pipeline for both backend and UI
@pipeline(enable_cache=False)
def auichat_full_deployment_pipeline():
    """
    Full deployment pipeline that deploys both the RAG backend to Cloud Run
    and the UI to Firebase Hosting.
    """
    # Run the preprocessing and indexing steps
    nodes_file = preprocess_data()
    index_status = create_and_store_index(nodes_file)
    validation_result = validate_qdrant_storage(after=[index_status])

    if not validation_result:
        print("❌ Qdrant storage validation failed. Stopping pipeline.")
        return

    # Deploy the RAG service to Cloud Run with vector search
    try:
        # Deploy to Cloud Run using our new step
        cloudrun_info = deploy_cloudrun_rag_service(
            preprocessed_nodes_path=nodes_file,
            collection_name=QDRANT_COLLECTION_NAME,
            after=[validation_result]
        )
        
        print(f"✅ RAG service deployed to Cloud Run!")
        
        # Deploy UI to Firebase Hosting
        firebase_info = deploy_ui_to_firebase(
            project_id=os.environ.get("PROJECT_ID", None),
            after=[cloudrun_info]
        )
        
        print("\n✅ Full deployment pipeline completed!")
        print("Your AUIChat application is now fully deployed to Google Cloud!")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        print("⚠️ Check the logs above for more details.")

if __name__ == "__main__":
    # Process environment variable for deployment type
    deployment_type = os.environ.get("AUICHAT_DEPLOYMENT", "").lower()
    
    # Process command line arguments
    if len(sys.argv) > 1:
        deployment_type = sys.argv[1].lower()
    
    # Activate appropriate stack based on deployment type
    if deployment_type in ["seldon", "vertex", "cloudrun", "firebaseui", "cloudrun-ui", "full"]:
        activate_gcp_stack()
    
    # Run the appropriate pipeline based on deployment type
    if deployment_type == "seldon":
        print("🚀 Running pipeline with Seldon deployment...")
        auichat_seldon_deployment_pipeline()
    elif deployment_type == "vertex":
        print("🚀 Running pipeline with Vertex AI deployment...")
        auichat_vertex_deployment_pipeline()
    elif deployment_type == "cloudrun":
        print("🚀 Running pipeline with Cloud Run RAG deployment...")
        auichat_cloudrun_rag_pipeline()
    elif deployment_type == "firebaseui":
        print("🚀 Running pipeline with Firebase UI deployment...")
        auichat_firebase_ui_pipeline()
    elif deployment_type == "cloudrun-ui":
        print("🚀 Running pipeline with Cloud Run UI deployment...")
        auichat_cloudrun_ui_pipeline()
    elif deployment_type == "full":
        print("🚀 Running full deployment pipeline (backend + UI)...")
        auichat_full_deployment_pipeline()
    elif deployment_type == "dev" or deployment_type == "test":
        print("🧪 Running development testing pipeline...")
        auichat_development_test_pipeline()
    else:
        print("🚀 Running pipeline without deployment step...")
        auichat_no_deploy_pipeline()
