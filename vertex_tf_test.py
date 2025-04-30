#!/usr/bin/env python3
# vertex_tf_test.py - Temporary test script for Vertex AI deployment with TensorFlow

import os
import sys
import logging
import json
import tempfile
import shutil
from typing import Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path so we can import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import the deployment function - using direct import to bypass ZenML step decorators
try:
    from rag_based_llm_auichat.src.workflows.vertex_deployment import deploy_model_to_vertex
except ImportError:
    logger.error("Failed to import deploy_model_to_vertex. Please check the module path.")
    sys.exit(1)


class MockStepArtifact:
    """Mock class to simulate a ZenML StepArtifact object"""
    def __init__(self, uri):
        self.uri = uri
    
    def __str__(self):
        return f"MockStepArtifact(uri={self.uri})"


def setup_environment():
    """Set up environment variables needed for deployment"""
    # Check if environment variables are already set
    project_id = os.environ.get("PROJECT_ID")
    region = os.environ.get("REGION")
    
    if not project_id:
        os.environ["PROJECT_ID"] = "deft-waters-458118-a3"  # Using the project ID from your scripts
        logger.info(f"Set PROJECT_ID to {os.environ['PROJECT_ID']}")
    
    if not region:
        os.environ["REGION"] = "us-central1"  # Using the region from your scripts
        logger.info(f"Set REGION to {os.environ['REGION']}")


def create_tensorflow_saved_model():
    """Create a simple TensorFlow SavedModel that Vertex AI can deploy"""
    try:
        import tensorflow as tf
        import numpy as np
        
        logger.info("Creating a simple TensorFlow model")
        
        # Create a very simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        # Create some dummy data and train the model for a single step
        x = np.random.random((10, 10))
        y = np.random.randint(0, 2, (10, 1))
        model.fit(x, y, epochs=1, verbose=0)
        
        # Create a temporary directory for the model
        model_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary model directory: {model_dir}")
        
        # Save the model in TensorFlow SavedModel format
        saved_model_path = os.path.join(model_dir, "saved_model")
        tf.saved_model.save(model, saved_model_path)
        
        logger.info(f"Saved TensorFlow model to {saved_model_path}")
        return model_dir
    
    except ImportError:
        logger.error("TensorFlow is not installed. Please install it with: pip install tensorflow")
        sys.exit(1)


def test_deployment(model_uri_type="string"):
    """Test Vertex AI deployment with the specified model URI type"""
    # Create the TensorFlow SavedModel
    base_model_dir = create_tensorflow_saved_model()
    
    # Prepare the model_uri according to the requested type
    if model_uri_type == "string":
        logger.info("Testing deployment with a string URI")
        model_uri = base_model_dir
    elif model_uri_type == "artifact":
        logger.info("Testing deployment with a mock StepArtifact")
        model_uri = MockStepArtifact(base_model_dir)
    elif model_uri_type == "dict":
        logger.info("Testing deployment with a dictionary")
        model_uri = {"uri": base_model_dir}
    else:
        raise ValueError(f"Unknown model_uri_type: {model_uri_type}")
    
    # Configure deployment parameters
    deployment_params = {
        "machine_type": "n1-standard-2",
        "min_replicas": 1,
        "max_replicas": 1,
        "accelerator_type": None,
        "accelerator_count": 0,
        "service_account": None,
    }
    
    try:
        # Attempt to deploy the model
        logger.info(f"Deploying model to Vertex AI with URI type: {model_uri_type}")
        result = deploy_model_to_vertex(model_uri=model_uri, **deployment_params)
        
        # Process and log the result
        if isinstance(result, dict) and 'endpoint_url' in result and result['endpoint_url'] != 'deployment_failed':
            logger.info("✅ Deployment successful!")
            logger.info(f"Endpoint URL: {result['endpoint_url']}")
            success = True
        else:
            logger.error("❌ Deployment failed")
            if isinstance(result, dict) and 'error' in result:
                logger.error(f"Error details: {result['error']}")
            success = False
        
        return success, result
    
    except Exception as e:
        logger.error(f"❌ Exception during deployment: {str(e)}")
        return False, {"error": str(e)}
    
    finally:
        # Clean up the temporary model directory
        try:
            if os.path.exists(base_model_dir):
                shutil.rmtree(base_model_dir)
                logger.info(f"Cleaned up temporary model directory: {base_model_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up directory {base_model_dir}: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Vertex AI deployment test with TensorFlow model")
    setup_environment()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test Vertex AI deployment")
    parser.add_argument("--type", choices=["string", "artifact", "dict"], 
                      default="string", help="Type of model URI to test")
    parser.add_argument("--dry-run", action="store_true", 
                      help="Prepare model but don't deploy")
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run mode - creating model but not deploying")
        model_dir = create_tensorflow_saved_model()
        logger.info(f"Model prepared at: {model_dir}")
        logger.info("To deploy manually, run the following command:")
        logger.info(f"python vertex_tf_test.py --type {args.type}")
    else:
        # Run the test with the specified URI type
        success, result = test_deployment(args.type)
        
        # Print final status
        if success:
            logger.info("✅ TEST PASSED - Deployment was successful")
        else:
            logger.info("❌ TEST FAILED - Deployment encountered errors")