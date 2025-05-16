#!/usr/bin/env python3
"""
Script to verify llama-index imports using the installed packages.
This will help us diagnose the import errors for VectorStoreIndex.
"""
import os
import sys
import logging
import importlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Checking installed llama-index packages...")

try:
    import pkg_resources
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    llama_packages = {k: v for k, v in installed_packages.items() if 'llama' in k.lower()}
    
    if llama_packages:
        logger.info("Installed llama packages:")
        for pkg, version in llama_packages.items():
            logger.info(f"- {pkg}: {version}")
    else:
        logger.warning("No llama-index packages found in the environment")
except Exception as e:
    logger.error(f"Error checking installed packages: {e}")

# Try to import specific modules to see what's available
logger.info("Checking for specific modules...")

modules_to_check = [
    "llama_index",
    "llama_index.core",
    "llama_index_core",
    "llama_index.retrievers",
    "llama_index.core.retrievers",
    "llama_index_core.retrievers",
    "llama_index.vector_stores.qdrant",
    "llama_index_vector_stores_qdrant"
]

for module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✅ Successfully imported {module_name}")
        
        # For llama_index check version
        if module_name == "llama_index" and hasattr(module, "__version__"):
            logger.info(f"   Version: {module.__version__}")
            
        # Check module content
        if hasattr(module, "__all__"):
            logger.info(f"   Exports: {module.__all__}")
    except ImportError as e:
        logger.warning(f"❌ Cannot import {module_name}: {e}")
    except Exception as e:
        logger.error(f"⚠️ Error checking {module_name}: {e}")

logger.info("Attempting imports...")

# Try import with main package structure first
try:
    # Test if the main llama-index package is available
    import llama_index
    logger.info(f"Successfully imported llama_index version {llama_index.__version__ if hasattr(llama_index, '__version__') else 'unknown'}")
    
    # Test if the core module is accessible through the main package
    try:
        from llama_index import VectorStoreIndex, Settings, StorageContext
        logger.info("Successfully imported core classes from llama_index")
    except ImportError as e:
        logger.warning(f"Could not import core classes from main package: {e}")
    
    # Try importing all required classes
    try:
        # First through the main package
        from llama_index.retrievers import BM25Retriever
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.huggingface import HuggingFaceLLM
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        logger.info("Successfully imported all required classes through main package")
    except ImportError as e:
        logger.warning(f"Some imports failed through main package: {e}")
        
        # Then try through the modular packages
        try:
            from llama_index.core import VectorStoreIndex, Settings, StorageContext
            logger.info("Successfully imported core classes through llama_index.core")
            
            # Try each import individually to identify specific issues
            try:
                from llama_index.core.node_parser import SentenceSplitter
                logger.info("✓ Imported SentenceSplitter")
            except ImportError as e:
                logger.warning(f"✗ Could not import SentenceSplitter: {e}")
                
            try:
                from llama_index.core.retrievers import BM25Retriever
                logger.info("✓ Imported BM25Retriever")
            except ImportError as e:
                logger.warning(f"✗ Could not import BM25Retriever: {e}")
                
            try:
                from llama_index.core.retrievers import VectorIndexRetriever
                logger.info("✓ Imported VectorIndexRetriever")
            except ImportError as e:
                logger.warning(f"✗ Could not import VectorIndexRetriever: {e}")
                
            try:
                from llama_index.core.retrievers import HybridRetriever
                logger.info("✓ Imported HybridRetriever")
            except ImportError as e:
                logger.warning(f"✗ Could not import HybridRetriever: {e}")
            
            try:
                from llama_index.core.query_engine import RetrieverQueryEngine
                logger.info("✓ Imported RetrieverQueryEngine")
            except ImportError as e:
                logger.warning(f"✗ Could not import RetrieverQueryEngine: {e}")
            
            try:
                from llama_index.core.response_synthesizers import CompactAndRefine
                logger.info("✓ Imported CompactAndRefine")
            except ImportError as e:
                logger.warning(f"✗ Could not import CompactAndRefine: {e}")
            
            try:
                from llama_index.core.postprocessor import SimilarityPostprocessor
                logger.info("✓ Imported SimilarityPostprocessor")
            except ImportError as e:
                logger.warning(f"✗ Could not import SimilarityPostprocessor: {e}")
            
            try:
                from llama_index.core.postprocessor.types import BaseNodePostprocessor
                logger.info("✓ Imported BaseNodePostprocessor")
            except ImportError as e:
                logger.warning(f"✗ Could not import BaseNodePostprocessor: {e}")
            
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                logger.info("✓ Imported HuggingFaceEmbedding")
            except ImportError as e:
                logger.warning(f"✗ Could not import HuggingFaceEmbedding: {e}")
            
            try:
                from llama_index.llms.huggingface import HuggingFaceLLM
                logger.info("✓ Imported HuggingFaceLLM")
            except ImportError as e:
                logger.warning(f"✗ Could not import HuggingFaceLLM: {e}")
            
            try:
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                logger.info("✓ Imported QdrantVectorStore")
            except ImportError as e:
                logger.warning(f"✗ Could not import QdrantVectorStore: {e}")
            
        except ImportError as e:
            logger.error(f"Failed to import through modular packages too: {e}")
            raise
except ImportError as e:
    logger.error(f"Failed to import llama_index: {e}")
    
    # Try direct import from core
    try:
        from llama_index_core import VectorStoreIndex, Settings, StorageContext
        logger.info("Successfully imported from llama_index_core")
    except ImportError as e2:
        logger.error(f"Failed to import from llama_index_core too: {e2}")
        raise

logger.info("✅ All import tests completed")
