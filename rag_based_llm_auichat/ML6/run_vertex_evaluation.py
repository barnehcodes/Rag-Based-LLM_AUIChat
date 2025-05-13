#!/usr/bin/env python3
"""
Standalone script to run RAG evaluation using Vertex AI.
This can be used outside the ZenML pipeline for ad-hoc evaluation.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML6.vertex_ai_evaluation import run_vertex_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description='Evaluate RAG system with Vertex AI')
    parser.add_argument('--endpoint', 
                        default="https://auichat-rag-qdrant-448245131663.us-central1.run.app/predict",
                        help='RAG endpoint URL')
    parser.add_argument('--project', 
                        default=os.environ.get("PROJECT_ID", "deft-waters-458118-a3"),
                        help='Google Cloud project ID')
    parser.add_argument('--region', 
                        default=os.environ.get("GCP_REGION", "us-central1"), 
                        help='Google Cloud region')
    parser.add_argument('--output', 
                        help='Output file for evaluation results (optional)')
    
    args = parser.parse_args()
    
    logger.info(f"Starting evaluation of RAG endpoint: {args.endpoint}")
    logger.info(f"Project ID: {args.project}, Region: {args.region}")
    
    # Run the evaluation
    results = run_vertex_evaluation(
        endpoint_url=args.endpoint,
        project_id=args.project,
        region=args.region
    )
    
    # Print summary
    print("\n=== RAG Evaluation Results ===")
    if "evaluation_results" in results and "average_metrics" in results["evaluation_results"]:
        metrics = results["evaluation_results"]["average_metrics"]
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("Evaluation failed or returned unexpected format")
    
    # Print robustness results
    print("\n=== Robustness Evaluation Results ===")
    if "robustness_results" in results and "robustness_score" in results["robustness_results"]:
        robustness_score = results["robustness_results"]["robustness_score"]
        print(f"Robustness Score: {robustness_score:.4f}")
        
        # Print detailed results
        test_count = results["robustness_results"]["adversarial_test_count"]
        correct_count = sum(1 for r in results["robustness_results"]["detailed_results"] if r["is_correct"])
        print(f"Adversarial Tests: {test_count}")
        print(f"Correctly Handled: {correct_count}")
        print(f"Incorrectly Handled: {test_count - correct_count}")
        
        # Print examples of incorrect handling (if any)
        incorrect_tests = [r for r in results["robustness_results"]["detailed_results"] if not r["is_correct"]]
        if incorrect_tests:
            print("\nExamples of incorrectly handled queries:")
            for i, test in enumerate(incorrect_tests[:2], 1):  # Show up to 2 examples
                print(f"  {i}. Question: {test['question']}")
                print(f"     Expected: {test['expected_behavior']}")
                print(f"     Actual: {test['actual_behavior']}")
    else:
        print("Robustness evaluation failed or returned unexpected format")
    
    # Print Vertex AI resources
    if "vertex_ai_model" in results:
        print(f"\nVertex AI Model: {results['vertex_ai_model']}")
    if "vertex_ai_job" in results:
        print(f"Vertex AI Evaluation Job: {results['vertex_ai_job']}")
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
        
        # Also save separate robustness evaluation report if robustness results are available
        if "robustness_results" in results:
            robustness_filename = args.output.replace('.json', '_robustness.json')
            with open(robustness_filename, 'w') as f:
                json.dump(results["robustness_results"], f, indent=2)
            print(f"Robustness results saved to: {robustness_filename}")
    
    # Return success
    return 0

if __name__ == "__main__":
    sys.exit(main())
