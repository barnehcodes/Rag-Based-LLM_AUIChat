#!/usr/bin/env python3
"""
Simple test script for verifying Vertex AI deployment with MLflow model URIs
"""
import os
import mlflow
from zenml.client import Client
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_model_uri_from_mlflow():
    """
    Get the most recent MLflow model URI from the tracking server
    """
    try:
        # Get the tracking URI
        tracking_uri = "/home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/mlruns/"
        print(f"MLflow tracking URI: {tracking_uri}")
        
        # List all runs in the current experiment
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=["0"])
        
        if not runs:
            print("❌ No MLflow runs found.")
            return None
            
        # Sort runs by start time (most recent first)
        sorted_runs = sorted(runs, key=lambda x: x.info.start_time, reverse=True)
        latest_run = sorted_runs[0]
        run_id = latest_run.info.run_id
        
        print(f"Found latest run: {run_id}")
        
        # Check if there's a model artifact in this run
        artifacts = client.list_artifacts(run_id)
        model_artifact = None
        
        for artifact in artifacts:
            if artifact.path == "model":
                model_artifact = artifact
                break
                
        if model_artifact:
            # Construct the model URI
            model_uri = f"runs:/{run_id}/model"
            print(f"✅ Found model URI: {model_uri}")
            return model_uri
        else:
            print(f"❌ No model artifact found in run {run_id}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting MLflow model URI: {str(e)}")
        return None

def main():
    """Main function to test deployment"""
    # Ensure GCP environment variables are set
    project_id = os.environ.get("PROJECT_ID")
    region = os.environ.get("REGION")
    
    if not project_id or not region:
        print("❌ PROJECT_ID and/or REGION environment variables are not set.")
        print("Please set them before running this script:")
        print("  export PROJECT_ID=your-project-id")
        print("  export REGION=your-region")
        return
    
    print(f"🔧 Using Google Cloud Project: {project_id} in region {region}")
    
    # Activate GCP stack
    activate_gcp_stack()
    
    # Get model URI from MLflow
    model_uri = get_model_uri_from_mlflow()
    
    if not model_uri:
        print("❌ Could not find a valid model URI.")
        return
    
    print("✅ Successfully found model URI for deployment")
    print(f"   URI: {model_uri}")
    
    # Import the Vertex deployment function
    try:
        from rag_based_llm_auichat.src.workflows.vertex_deployment import deploy_model_to_vertex
        
        # Deploy model to Vertex AI
        print("🚀 Deploying model to Vertex AI...")
        result = deploy_model_to_vertex(
            model_uri=model_uri,
            machine_type="n1-standard-2",
            min_replicas=1,
            max_replicas=1
        )
        
        # Check deployment status
        if isinstance(result, dict):
            if result.get("endpoint_url") == "deployment_failed":
                print(f"❌ Deployment failed: {result.get('error')}")
            else:
                print("✅ Model deployed successfully!")
                print(f"   Endpoint URL: {result.get('endpoint_url')}")
        else:
            print(f"⚠️ Unexpected result type: {type(result)}")
    except ImportError:
        print("❌ Could not import deploy_model_to_vertex. Check your Python path.")
    except Exception as e:
        print(f"❌ Error during deployment: {str(e)}")

if __name__ == "__main__":
    main()