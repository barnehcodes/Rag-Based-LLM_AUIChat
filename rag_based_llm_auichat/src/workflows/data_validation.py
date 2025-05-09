from zenml import step
from zenml.logger import get_logger
import os
import pickle

logger = get_logger(__name__)

@step
def validate_processed_data_step(nodes_file_path: str) -> bool:
    """
    Validates the preprocessed data (nodes file) exists and has the expected structure.
    
    Args:
        nodes_file_path: Path to the preprocessed nodes file
        
    Returns:
        Boolean indicating if validation passed
    """
    logger.info(f"Validating preprocessed data at: {nodes_file_path}")
    
    # Check if file exists
    if not os.path.exists(nodes_file_path):
        logger.error(f"Preprocessed nodes file not found at: {nodes_file_path}")
        return False
    
    # Check if file is not empty
    if os.path.getsize(nodes_file_path) == 0:
        logger.error(f"Preprocessed nodes file is empty: {nodes_file_path}")
        return False
    
    # Try to load the file to verify it's a valid pickle file with nodes
    try:
        with open(nodes_file_path, 'rb') as f:
            nodes = pickle.load(f)
        
        # Check if it's a list/iterable
        try:
            node_count = len(nodes)
            logger.info(f"Successfully validated {node_count} nodes in {nodes_file_path}")
            
            # Optional: add more specific validation if needed
            # For example, check if nodes have expected attributes
            if node_count > 0:
                sample_node = nodes[0]
                if not hasattr(sample_node, 'text') and not hasattr(sample_node, 'get_text'):
                    logger.warning("Nodes may not have the expected structure (missing 'text' attribute)")
                    # Not failing validation for this, just a warning
            
            return True
            
        except TypeError:
            logger.error(f"Preprocessed nodes file does not contain an iterable: {nodes_file_path}")
            return False
            
    except (pickle.UnpicklingError, EOFError):
        logger.error(f"Preprocessed nodes file is not a valid pickle file: {nodes_file_path}")
        return False
    except Exception as e:
        logger.error(f"Error validating preprocessed nodes file: {e}")
        return False