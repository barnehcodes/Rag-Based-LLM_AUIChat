#!/usr/bin/env python3
"""
Test Version B of the RAG System Locally
---------------------------------------
This script tests the Version B RAG system locally without deploying to Cloud Run,
allowing you to ensure that the advanced features work correctly:

1. Hybrid Retrieval (vector + BM25)
2. Query Reformulation
3. Advanced Re-ranking with cross-encoders
4. Multi-hop RAG architecture
5. Improved prompting
6. Better generation parameters
"""

import os
import sys
import pickle
import logging
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the necessary components from Version B
from sentence_transformers import CrossEncoder

# Handle compatibility with different llama-index versions
try:
    # Try the current structure first with .core namespace
    from llama_index.core import VectorStoreIndex, Settings, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.core.retrievers import BM25Retriever
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.retrievers import HybridRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import CompactAndRefine
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    logger.info("Using llama_index.core imports")
except ImportError:
    # Fall back to older import structure
    from llama_index import VectorStoreIndex, Settings, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.retrievers import BM25Retriever
    from llama_index.retrievers import VectorIndexRetriever
    from llama_index.retrievers import HybridRetriever
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.response_synthesizers import CompactAndRefine
    from llama_index.postprocessor.types import BaseNodePostprocessor
    logger.info("Using legacy llama_index imports")
from transformers import T5Tokenizer, T5ForConditionalGeneration

# QueryReformulator class from Version B script
class QueryReformulator:
    """
    Uses a T5-based model to reformulate user queries to improve retrieval.
    Expands acronyms, adds specific terms, and makes queries more precise.
    """
    def __init__(self):
        logger.info("Initializing query reformulation model...")
        model_name = "google/flan-t5-small"  # Smaller model for query reformulation
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("Query reformulation model loaded.")
        
        # Add specific context about AUI
        self.context_prefix = "Al Akhawayn University is a university in Ifrane, Morocco. "
        
        # Common acronyms in AUI context
        self.acronyms = {
            "AUI": "Al Akhawayn University in Ifrane",
            "SBA": "School of Business Administration",
            "SSE": "School of Science and Engineering",
            "SHSS": "School of Humanities and Social Sciences",
            "PiP": "Partners in Progress"
        }
        
    def expand_acronyms(self, query: str) -> str:
        """Expand known acronyms in the query"""
        expanded = query
        for acronym, expansion in self.acronyms.items():
            # Only replace if the acronym appears as a whole word
            expanded = expanded.replace(f" {acronym} ", f" {expansion} ")
            if expanded.startswith(f"{acronym} "):
                expanded = f"{expansion} " + expanded[len(acronym)+1:]
            if expanded.endswith(f" {acronym}"):
                expanded = expanded[:-len(acronym)-1] + f" {expansion}"
            if expanded == acronym:
                expanded = expansion
        return expanded
    
    def reformulate(self, query: str) -> str:
        """
        Reformulate the query to make it more specific and retrievable.
        """
        # Step 1: Expand any acronyms in the query
        expanded_query = self.expand_acronyms(query)
        
        # Step 2: Prepare prompt for the model
        prompt = f"Rewrite this question to make it more specific and detailed for a university information retrieval system. Original question: {expanded_query}"
        
        # Step 3: Generate reformulation
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=150,
            temperature=0.2,
            num_return_sequences=1
        )
        reformulated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If the model output is too different or doesn't make sense, use the expanded query
        if len(reformulated.split()) < 3 or len(reformulated) / len(expanded_query) < 0.5:
            logger.warning(f"Query reformulation failed, using expanded query: {expanded_query}")
            return expanded_query
        
        logger.info(f"Original: '{query}' → Reformulated: '{reformulated}'")
        return reformulated

# CrossEncoderReranker class from Version B script
class CrossEncoderReranker(BaseNodePostprocessor):
    """
    Uses a cross-encoder model to rerank retrieved documents for better precision.
    """
    def __init__(self, top_n: int = 5):
        logger.info("Initializing cross-encoder reranker...")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.top_n = top_n
        logger.info("Cross-encoder reranker initialized.")
    
    def postprocess_nodes(self, nodes, query_str):
        if not nodes:
            return []
        
        # Create pairs of (query, document text) for scoring
        pairs = [(query_str, node.get_text()) for node in nodes]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Assign new scores to nodes
        for node, score in zip(nodes, scores):
            node.score = float(score)  # Update node score with cross-encoder score
        
        # Sort by new score and take top_n
        reranked_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)[:self.top_n]
        
        return reranked_nodes

# MultiHopQueryEngine class from Version B script
class MultiHopQueryEngine:
    """
    Implements a multi-hop RAG architecture for complex questions.
    First retrieval gets basic information, which is used for a second retrieval step.
    """
    def __init__(self, retriever, llm, similarity_top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        
        # Synthesizer for compact and more efficient responses
        self.response_synthesizer = CompactAndRefine(
            llm=self.llm,
            verbose=True,
        )
        
        # Standard query engine for first hop
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
        )
    
    def _needs_second_hop(self, query: str, first_hop_nodes) -> bool:
        """Determine if we need a second hop based on first results"""
        # Simple heuristic: if question is long or contains specific keywords
        complexity_indicators = ["specific", "details", "explain", "how", "requirements", 
                               "what are the", "tell me about", "prerequisites"]
        
        query_lower = query.lower()
        is_complex = (
            len(query.split()) >= 8 or
            any(indicator in query_lower for indicator in complexity_indicators)
        )
        
        # If we don't have enough context from first hop
        has_insufficient_context = len(first_hop_nodes) < 2
        
        return is_complex and not has_insufficient_context
    
    def _generate_follow_up_query(self, query: str, first_hop_response: str) -> str:
        """Generate a follow-up query based on the first response"""
        prompt = f"""
        Based on the following question and initial information, generate a follow-up question to get more specific details:
        
        Original Question: {query}
        
        Initial Information: {first_hop_response}
        
        Follow-up Question:
        """
        
        # Get the response
        response = self.llm.complete(prompt)
        follow_up = response.text.strip()
        
        if not follow_up:
            follow_up = f"Tell me more specific details about {query}"
            
        logger.info(f"Follow-up query: {follow_up}")
        return follow_up
    
    def query(self, query_str: str) -> tuple:
        """
        Execute multi-hop RAG query process.
        Returns both the response and nodes used for generation.
        """
        # First hop retrieval and response generation
        first_hop_nodes = self.retriever.retrieve(query_str)
        
        if not self._needs_second_hop(query_str, first_hop_nodes):
            # For simple queries, just use standard query engine
            response = self.query_engine.query(query_str)
            return response, first_hop_nodes
        
        # Generate intermediate answer from first hop
        first_response = self.response_synthesizer.synthesize(
            query=query_str,
            nodes=first_hop_nodes
        )
        
        # Generate follow-up query based on first response
        follow_up_query = self._generate_follow_up_query(
            query_str, 
            first_response.response
        )
        
        # Second hop retrieval
        second_hop_nodes = self.retriever.retrieve(follow_up_query)
        
        # Combine unique nodes from both hops
        all_node_ids = set()
        combined_nodes = []
        
        for node in first_hop_nodes + second_hop_nodes:
            if node.node_id not in all_node_ids:
                all_node_ids.add(node.node_id)
                combined_nodes.append(node)
        
        # Final response generation with all collected evidence
        final_response = self.response_synthesizer.synthesize(
            query=query_str,
            nodes=combined_nodes
        )
        
        return final_response, combined_nodes

def setup_version_b_components(nodes_path):
    """
    Set up all the components for Version B RAG system.
    
    Args:
        nodes_path: Path to preprocessed nodes pickle file
    
    Returns:
        Dict containing all the Version B components
    """
    # Load nodes
    logger.info(f"Loading preprocessed nodes from {nodes_path}")
    with open(nodes_path, 'rb') as f:
        nodes = pickle.load(f)
    logger.info(f"Loaded {len(nodes)} nodes")
    
    # Initialize models
    logger.info("Loading LLM and embedding models...")
    llm = HuggingFaceLLM(
        model_name="google/gemma-2b",
        tokenizer_name="google/gemma-2b",
        context_window=2048,
        max_new_tokens=512,
        model_kwargs={"temperature": 0.3},  # Lower temperature for better factuality
        generate_kwargs={"do_sample": True}
    )
    
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Update settings
    Settings.llm = llm
    Settings.embed_model = embedding_model
    
    # Create index
    logger.info("Creating index from nodes...")
    index = VectorStoreIndex(nodes)
    vector_store = index.vector_store
    
    # Set up storage context and vector index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    
    # Set up retrievers
    logger.info("Setting up hybrid retrieval system...")
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=7
    )
    
    # Create BM25 retriever
    all_nodes = vector_index.docstore.docs.values()
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=list(all_nodes),
        similarity_top_k=7
    )
    
    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,  # Will be reranked later
        weights=[0.7, 0.3]
    )
    
    # Set up query reformulator
    logger.info("Setting up query reformulation...")
    query_reformulator = QueryReformulator()
    
    # Set up cross-encoder reranker
    logger.info("Setting up cross-encoder reranker...")
    cross_encoder_reranker = CrossEncoderReranker(top_n=5)
    
    # Set up multi-hop query engine
    logger.info("Setting up multi-hop query engine...")
    multi_hop_engine = MultiHopQueryEngine(
        retriever=hybrid_retriever,
        llm=llm,
        similarity_top_k=5
    )
    
    return {
        "llm": llm,
        "embedding_model": embedding_model,
        "index": index,
        "hybrid_retriever": hybrid_retriever,
        "query_reformulator": query_reformulator,
        "cross_encoder_reranker": cross_encoder_reranker,
        "multi_hop_engine": multi_hop_engine
    }

def process_query(components, query):
    """
    Process a query using the Version B RAG system.
    
    Args:
        components: Dict containing Version B components
        query: User query string
    
    Returns:
        Dict with response and processing details
    """
    logger.info(f"Processing query: '{query}'")
    
    # Step 1: Query reformulation
    reformulated_query = components["query_reformulator"].reformulate(query)
    logger.info(f"Reformulated query: '{reformulated_query}'")
    
    # Step 2: Multi-hop RAG query process
    response, retrieved_nodes = components["multi_hop_engine"].query(reformulated_query)
    logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
    
    # Step 3: Further reranking of the retrieved nodes
    reranked_nodes = components["cross_encoder_reranker"].postprocess_nodes(
        retrieved_nodes,
        reformulated_query
    )
    logger.info(f"Reranked to {len(reranked_nodes)} top nodes")
    
    # Format sources
    sources = []
    for node in reranked_nodes:
        if hasattr(node, "metadata") and node.metadata:
            source = {
                "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "score": float(node.score) if hasattr(node, "score") else 0.0
            }
            
            # Add source file information if available
            if "file_name" in node.metadata:
                source["file_name"] = node.metadata["file_name"]
            elif "source" in node.metadata:
                source["file_name"] = node.metadata["source"]
            
            sources.append(source)
    
    result = {
        "original_query": query,
        "reformulated_query": reformulated_query,
        "response": response.response,
        "sources": sources
    }
    
    return result

def main():
    """Main entry point for the script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Version B of the RAG system locally."
    )
    parser.add_argument(
        "--preprocessed-nodes",
        default="/home/barneh/Rag-Based-LLM_AUIChat/preprocessed_nodes.pkl",
        help="Path to the preprocessed nodes pickle file"
    )
    parser.add_argument(
        "--query",
        default="What are AUI's admission requirements?",
        help="Query to test the RAG system with"
    )
    
    args = parser.parse_args()
    
    # Make sure the nodes file exists
    if not os.path.exists(args.preprocessed_nodes):
        logger.error(f"Preprocessed nodes file not found: {args.preprocessed_nodes}")
        return 1
    
    try:
        # Set up Version B components
        logger.info("Setting up Version B RAG components...")
        components = setup_version_b_components(args.preprocessed_nodes)
        
        # Process query
        logger.info("Processing test query...")
        result = process_query(components, args.query)
        
        # Print results in a nice format
        print("\n" + "=" * 80)
        print("VERSION B RAG SYSTEM TEST RESULTS")
        print("=" * 80)
        print(f"Original Query: {result['original_query']}")
        print(f"Reformulated Query: {result['reformulated_query']}")
        print("-" * 80)
        print("RESPONSE:")
        print(result['response'])
        print("-" * 80)
        print("TOP SOURCES:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source.get('file_name', 'Unknown source')}")
            print(f"   Score: {source['score']:.4f}")
            print(f"   Text: {source['text'][:100]}...")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error testing Version B RAG system: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
