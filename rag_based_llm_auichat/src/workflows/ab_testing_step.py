"""
ZenML step for A/B testing between two RAG endpoints.
This step can be used in pipelines to automatically compare current and modified RAG systems.
"""

from zenml import step
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)

@step
def ab_testing_step(
    current_endpoint: str,  # Current production endpoint
    modified_endpoint: str,  # Modified endpoint with improvements
    project_id: str = os.environ.get("PROJECT_ID", "deft-waters-458118-a3"),
    region: str = os.environ.get("GCP_REGION", "us-central1"),
    detailed_logging: bool = True
) -> Dict[str, Any]:
    """
    Compare two RAG system endpoints for performance evaluation.
    This ZenML step performs A/B testing between current and modified RAG systems.
    
    Args:
        current_endpoint: URL of the current production RAG endpoint (System A)
        modified_endpoint: URL of the modified RAG endpoint (System B)
        project_id: Google Cloud project ID
        region: Google Cloud region
        detailed_logging: Whether to log detailed comparison info
        
    Returns:
        Dictionary containing comparison results
    """
    # Import here to avoid issues with ZenML serialization
    from ML6.ab_testing import compare_rag_systems, TEST_DATA
    
    # Ensure endpoints end with /predict
    if not current_endpoint.endswith("/predict"):
        current_endpoint = f"{current_endpoint}/predict"
        
    if not modified_endpoint.endswith("/predict"):
        modified_endpoint = f"{modified_endpoint}/predict"
    
    logger.info(f"Starting A/B testing:")
    logger.info(f"  Current system (A): {current_endpoint}")
    logger.info(f"  Modified system (B): {modified_endpoint}")
    
    # Run the comparison
    results = compare_rag_systems(
        endpoint_a=current_endpoint,
        endpoint_b=modified_endpoint,
        project_id=project_id,
        store_in_gcs=True
    )
    
    # Log summary results
    winner = results["overall_winner"]
    system_a_wins = results["system_a"]["wins"]
    system_b_wins = results["system_b"]["wins"]
    ties = results["ties"]
    total = results["total_questions"]
    
    logger.info(f"A/B testing complete - Overall winner: System {winner}")
    logger.info(f"System A wins: {system_a_wins}/{total} questions")
    logger.info(f"System B wins: {system_b_wins}/{total} questions")
    logger.info(f"Ties: {ties}/{total} questions")
    
    # Log average metrics
    logger.info("Average metrics comparison:")
    metrics_a = results["system_a"]["average_metrics"]
    metrics_b = results["system_b"]["average_metrics"]
    
    for metric in sorted(metrics_a.keys()):
        if metric in metrics_b:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]
            diff = value_b - value_a
            change = "→" if abs(diff) < 0.01 else "↑" if diff > 0 else "↓"
            logger.info(f"  {metric}: {value_a:.4f} → {value_b:.4f} {change} ({diff:+.4f})")
    
    # Log detailed comparison if requested
    if detailed_logging and len(results["detailed_results"]) > 0:
        # Take a sample result to show the difference
        sample = results["detailed_results"][0]
        logger.info("\nSample comparison for question:")
        logger.info(f"  Question: {sample['question']}")
        logger.info(f"  System A answer: {sample['system_a']['answer'][:100]}...")
        logger.info(f"  System B answer: {sample['system_b']['answer'][:100]}...")
        logger.info(f"  Winner: System {sample['winner']}")
    
    return results
