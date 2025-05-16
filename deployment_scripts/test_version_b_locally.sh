#!/bin/bash
# Script to test the Version B app code locally without deploying to Cloud Run

set -e

# Determine project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Create a temporary directory
TMP_DIR=$(mktemp -d)
echo "📁 Created temporary directory: ${TMP_DIR}"

# Cleanup function
cleanup() {
  echo "🧹 Cleaning up temporary directory"
  rm -rf "${TMP_DIR}"
}

# Register cleanup on exit
trap cleanup EXIT

# Copy necessary files
echo "📂 Preparing application files..."
cp "${PROJECT_ROOT}/rag_based_llm_auichat/preprocessed_nodes.pkl" "${TMP_DIR}/"

# Extract the app code from the main script
echo "📄 Extracting app code..."
SCRIPT_PATH="${SCRIPT_DIR}/deploy_version_b_cloudrun.sh"
sed -n '/^cat > "\${TMP_DIR}\/app.py" << '\''EOF'\''$/,/^EOF$/p' "$SCRIPT_PATH" | sed '1d;$d' > "${TMP_DIR}/app.py"

# Create a minimal requirements file
cat > "${TMP_DIR}/requirements.txt" << 'EOF'
# Core API dependencies
fastapi>=0.95.1
uvicorn>=0.22.0
qdrant-client>=1.1.1

# llama-index packages with consistent version
llama-index==0.12.35  # Use the main package that's already installed
llama-index-core==0.12.35  # Match your installed version
llama-index-embeddings-huggingface>=0.1.0  # More flexible version
llama-index-llms-huggingface>=0.1.0  # More flexible version
llama-index-vector-stores-qdrant>=0.1.0  # More flexible version
llama-index-readers-file>=0.1.0  # More flexible version
llama-index-retrievers-bm25>=0.1.0  # More flexible version

# Machine learning and NLP dependencies
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
python-multipart>=0.0.5
typing-extensions>=4.0.0

# For handling network issues during installation
pip-tools>=6.0.0
retry>=0.9.0
EOF

echo "📦 Setting up Python environment..."
cd "${TMP_DIR}"

# Check if pip is available
if ! command -v pip &> /dev/null; then
  echo "❌ ERROR: pip is not installed or not in PATH"
  exit 1
fi

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements with retry for network issues
pip install pip-tools retry
python -c "
import retry
import subprocess
import sys

@retry.retry(tries=5, delay=2, backoff=2)
def install_requirements():
    print('Installing requirements with retry mechanism...')
    result = subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=False)
    if result.returncode != 0:
        print('Failed to install requirements, will retry...')
        raise Exception('Installation failed')
    return True

try:
    install_requirements()
    print('Requirements installed successfully.')
except Exception as e:
    print(f'Failed to install requirements after multiple attempts: {e}')
    sys.exit(1)
"

# First, check for import errors
echo "🔍 Validating imports..."
cat > "${TMP_DIR}/validate_imports.py" << 'EOF'
"""Script to validate llama-index imports"""
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from llama_index.core import VectorStoreIndex, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import BM25Retriever
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.retrievers import HybridRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import CompactAndRefine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    logger.info("✅ All imports successful!")
    sys.exit(0)
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)
EOF

python "${TMP_DIR}/validate_imports.py"
VALIDATE_RESULT=$?

if [ $VALIDATE_RESULT -ne 0 ]; then
    echo "❌ Import validation failed. Please check the requirements and imports."
    exit 1
fi

echo "🚀 Starting local server..."
echo "Press Ctrl+C to stop the server"
PREPROCESSED_NODES="${TMP_DIR}/preprocessed_nodes.pkl" uvicorn app:app --host 0.0.0.0 --port 8080
