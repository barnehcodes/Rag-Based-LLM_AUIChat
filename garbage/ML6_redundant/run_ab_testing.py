#!/usr/bin/env python3
"""
Standalone script to run A/B testing between two RAG endpoints.
This is useful for ad-hoc comparisons between current and improved RAG systems.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def print_comparison_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of A/B testing results.
    
    Args:
        results: The comparison results dictionary
    """
    print("\n" + "="*80)
    print(f"                      RAG SYSTEM A/B TEST RESULTS")
    print("="*80)
    
    print(f"\nOVERALL WINNER: SYSTEM {results['overall_winner']}")
    print(f"\nSystem A wins: {results['system_a']['wins']}/{results['total_questions']} questions")
    print(f"System B wins: {results['system_b']['wins']}/{results['total_questions']} questions")
    print(f"Ties: {results['ties']}/{results['total_questions']} questions")
    
    print("\n" + "-"*80)
    print("AVERAGE METRICS")
    print("-"*80)
    
    metrics_a = results['system_a']['average_metrics']
    metrics_b = results['system_b']['average_metrics']
    
    # Calculate which metrics improved
    metrics_improved = {}
    for metric in metrics_a.keys():
        if metric in metrics_b:
            diff = metrics_b[metric] - metrics_a[metric]
            metrics_improved[metric] = diff
    
    # Print metrics side by side with comparison
    print(f"{'Metric':<20} {'System A':<10} {'System B':<10} {'Change':<10}")
    print("-" * 50)
    
    for metric in sorted(metrics_a.keys()):
        if metric in metrics_b:
            value_a = metrics_a[metric]
            value_b = metrics_b[metric]
            diff = metrics_improved[metric]
            
            # Format the difference with + or - sign and arrows
            if abs(diff) < 0.001:
                diff_str = "  ―"  # No significant change
            else:
                arrow = "↑" if diff > 0 else "↓"
                color_code = "\033[92m" if diff > 0 else "\033[91m"  # Green for improvement, red for regression
                reset_code = "\033[0m"
                diff_str = f"{color_code}{arrow} {abs(diff):.4f}{reset_code}"
                
            print(f"{metric:<20} {value_a:<10.4f} {value_b:<10.4f} {diff_str}")
    
    print("\n" + "-"*80)
    
    # If results were stored in GCS
    if "gcs_path" in results:
        print(f"\nDetailed results stored at: {results['gcs_path']}")
    
    print("\n" + "="*80)

def main():
    """Command-line entry point for A/B testing"""
    parser = argparse.ArgumentParser(description='Run A/B testing between two RAG endpoints')
    
    parser.add_argument("--endpoint-a", required=True,
                       help="URL of the current RAG endpoint (System A)")
    parser.add_argument("--endpoint-b", required=True,
                       help="URL of the modified RAG endpoint (System B)")
    parser.add_argument("--project", default=os.environ.get("PROJECT_ID", "deft-waters-458118-a3"),
                       help="Google Cloud project ID")
    parser.add_argument("--output", help="Save results to a local JSON file")
    parser.add_argument("--no-gcs", action="store_true", 
                       help="Don't store results in Google Cloud Storage")
    
    args = parser.parse_args()
    
    from ML6.ab_testing import compare_rag_systems
    
    # Run the A/B testing
    try:
        results = compare_rag_systems(
            endpoint_a=args.endpoint_a,
            endpoint_b=args.endpoint_b,
            project_id=args.project,
            store_in_gcs=not args.no_gcs
        )
        
        # Print formatted summary
        print_comparison_summary(results)
        
        # Save to local file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0
    except Exception as e:
        logger.error(f"Error running A/B testing: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
