"""
ZenML step for Vertex AI evaluation of the RAG system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from zenml import step
from zenml.logger import get_logger

from ML6.vertex_ai_evaluation import run_vertex_evaluation

logger = get_logger(__name__)

@step
def vertex_ai_evaluation_step(
    cloud_run_deployment_info: Dict[str, Any],
    project_id: str,
    region: str
) -> Dict[str, Any]:
    """
    ZenML step to evaluate the RAG model using Vertex AI.
    
    Args:
        cloud_run_deployment_info: Dictionary containing information about the deployed Cloud Run service
        project_id: Google Cloud project ID
        region: Google Cloud region
        
    Returns:
        Dictionary containing evaluation results and Vertex AI resources
    """
    # Extract endpoint URL from deployment info
    if not cloud_run_deployment_info or "service_url" not in cloud_run_deployment_info:
        raise ValueError("Invalid deployment info: missing service_url")
    
    endpoint_url = cloud_run_deployment_info["service_url"]
    logger.info(f"Evaluating RAG endpoint using Vertex AI: {endpoint_url}")
    
    # Run the evaluation
    results = run_vertex_evaluation(
        endpoint_url=endpoint_url,
        project_id=project_id,
        region=region
    )
    
    # Log results
    if "evaluation_results" in results and "average_metrics" in results["evaluation_results"]:
        metrics = results["evaluation_results"]["average_metrics"]
        logger.info("=== RAG Evaluation Results ===")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    return results
