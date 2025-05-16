#!/usr/bin/env python3
"""
Standalone A/B testing pipeline for AUIChat RAG system.
This pipeline compares two RAG endpoints to determine which performs better.
"""

import os
import sys
import logging
import argparse
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

# Add project root to PATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Import the A/B testing step
from src.workflows.ab_testing_step import ab_testing_step

# Configure logging
logger = get_logger(__name__)

# Define constants
PROJECT_ID = os.environ.get("PROJECT_ID", "deft-waters-458118-a3")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")

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

# -----------------------------------------------------------------------------
# A/B Testing Pipeline
# -----------------------------------------------------------------------------
@pipeline(name="AUICHAT_AB_TESTING_PIPELINE", enable_cache=False)
def auichat_ab_testing_pipeline(
    current_endpoint: str,
    modified_endpoint: str
):
    """
    Pipeline to run A/B testing between current and modified RAG endpoints.
    
    Args:
        current_endpoint: URL of the current production RAG endpoint
        modified_endpoint: URL of the modified/improved RAG endpoint
    """
    logger.info("🚀 Starting AUIChat A/B Testing Pipeline...")
    activate_gcp_stack()  # Activate GCP stack for cloud resources
    
    # Run A/B testing
    comparison_results = ab_testing_step(
        current_endpoint=current_endpoint,
        modified_endpoint=modified_endpoint,
        project_id=PROJECT_ID,
        region=GCP_REGION,
        detailed_logging=True
    )
    
    logger.info("✅ AUIChat A/B Testing Pipeline completed.")
    return comparison_results

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    """Command-line entry point for the pipeline"""
    parser = argparse.ArgumentParser(description='Run A/B testing pipeline for AUIChat')
    
    parser.add_argument("--current-endpoint", required=True,
                        help="URL of the current production RAG endpoint")
    parser.add_argument("--modified-endpoint", required=True,
                        help="URL of the modified RAG endpoint")
    
    args = parser.parse_args()
    
    # Run the pipeline
    logger.info(f"Running A/B testing pipeline between:")
    logger.info(f"  Current endpoint: {args.current_endpoint}")
    logger.info(f"  Modified endpoint: {args.modified_endpoint}")
    
    auichat_ab_testing_pipeline(
        current_endpoint=args.current_endpoint,
        modified_endpoint=args.modified_endpoint
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
