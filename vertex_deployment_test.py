#!/usr/bin/env python3
# vertex_deployment_test.py - Temporary test script for Vertex AI deployment functionality

import os
import sys
import logging
import json
import tempfile
import pickle
import numpy as np
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path so we can import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import the deployment function - using direct import to bypass ZenML step decorators
from rag_based_llm_auichat.src.workflows.vertex_deployment import deploy_model_to_vertex

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


def create_mock_model():
    """Create a simple mock model that can be saved as a pickle file"""
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple random forest model
        model = RandomForestClassifier(n_estimators=2, max_depth=2)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model.fit(X, y)
        logger.info("Created a simple scikit-learn model for testing")
        return model
    except ImportError:
        # Create a very simple mock model class if scikit-learn is not available
        class MockModel:
            def predict(self, data):
                return [1] * len(data)
                
        logger.info("Created a simple mock model for testing")
        return MockModel()


def create_valid_model_directory():
    """Create a temporary directory with valid model files required by Vertex AI"""
    # Create a temporary directory
    model_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary model directory: {model_dir}")
    
    # Create a simple model and save it
    model = create_mock_model()
    model_file = os.path.join(model_dir, "model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved mock model to {model_file}")
    
    # Create a stub MLflow model file
    mlflow_file = os.path.join(model_dir, "MLmodel")
    with open(mlflow_file, "w") as f:
        f.write("""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.10.12
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.3.0
mlflow_version: 2.4.1
model_uuid: 7f2eec33a7e248e4af40ca8383b5dafb
run_id: temporary_test_model
utc_time_created: '2025-04-27 21:15:00.000000'
""")
    logger.info(f"Created MLmodel file at {mlflow_file}")
    
    # Create conda.yaml and python_env.yaml files often required by MLflow
    conda_file = os.path.join(model_dir, "conda.yaml")
    with open(conda_file, "w") as f:
        f.write("""channels:
- conda-forge
dependencies:
- python=3.10.12
- pip<=23.1.2
- pip:
  - mlflow<3,>=2.4
  - cloudpickle==2.2.1
  - scikit-learn==1.3.0
name: mlflow-env
""")

    py_env_file = os.path.join(model_dir, "python_env.yaml")
    with open(py_env_file, "w") as f:
        f.write("""python: 3.10.12
build_dependencies:
- pip<=23.1.2
- setuptools==68.0.0
- wheel==0.40.0
dependencies:
- mlflow<3,>=2.4
- cloudpickle==2.2.1 
- scikit-learn==1.3.0
""")
    
    # Create a requirements.txt file
    req_file = os.path.join(model_dir, "requirements.txt")
    with open(req_file, "w") as f:
        f.write("""mlflow<3,>=2.4
cloudpickle==2.2.1
scikit-learn==1.3.0
""")
    
    return model_dir


def test_with_string_uri():
    """Test deployment with a string URI"""
    logger.info("Testing deployment with a string URI")
    model_uri = create_valid_model_directory()
    
    try:
        result = deploy_model_to_vertex(
            model_uri=model_uri,
            machine_type="n1-standard-2",
            min_replicas=1,
            max_replicas=1
        )
        logger.info(f"Deployment result: {result}")
        return True
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


def test_with_mock_step_artifact():
    """Test deployment with a mock StepArtifact"""
    logger.info("Testing deployment with a mock StepArtifact")
    model_uri = MockStepArtifact(create_valid_model_directory())
    
    try:
        result = deploy_model_to_vertex(
            model_uri=model_uri,
            machine_type="n1-standard-2",
            min_replicas=1,
            max_replicas=1
        )
        logger.info(f"Deployment result: {result}")
        return True
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


def test_with_dict():
    """Test deployment with a dictionary containing URI"""
    logger.info("Testing deployment with a dictionary")
    model_uri = {"uri": create_valid_model_directory()}
    
    try:
        result = deploy_model_to_vertex(
            model_uri=model_uri,
            machine_type="n1-standard-2",
            min_replicas=1,
            max_replicas=1
        )
        logger.info(f"Deployment result: {result}")
        return True
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting Vertex AI deployment test")
    setup_environment()
    
    # Run tests with different input types
    test_methods = [
        test_with_string_uri,
        test_with_mock_step_artifact,
        test_with_dict
    ]
    
    # Allow specifying a specific test via command line
    if len(sys.argv) > 1 and sys.argv[1] in ["string", "artifact", "dict"]:
        if sys.argv[1] == "string":
            test_methods = [test_with_string_uri]
        elif sys.argv[1] == "artifact":
            test_methods = [test_with_mock_step_artifact]
        else:
            test_methods = [test_with_dict]
    
    results = []
    for test_method in test_methods:
        logger.info(f"Running test: {test_method.__name__}")
        success = test_method()
        results.append((test_method.__name__, success))
    
    # Print summary
    logger.info("Test Summary:")
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} - {name}")