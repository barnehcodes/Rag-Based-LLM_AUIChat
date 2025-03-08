# ML Pipeline Development - From a Monolith to a Pipeline (Milestones 3)

## 2.1 Ensuring ML Pipeline Reproducibility (Milestone 3)

### ✅ Ensuring Pipeline Reproducibility

- **Project Setup and Reproducible Environment**
  - Established reproducibility through Python virtual environments managed by `venv` and documented dependencies via `requirements.txt`.

- **Data Versioning:**
  - Implemented DVC (Data Version Control) to version-control dataset files, document chunks, and embeddings.
  - Integrated DVC with Git to track changes in datasets separately from source code.

- **Machine Learning Pipeline Setup:**
  - Adopted ZenML to orchestrate the ML pipeline, ensuring modularity, reproducibility, and scalability.
  - Defined clearly structured pipeline steps using ZenML, enabling easy tracking and reproducibility.

## 2.2 Pipeline Components (Milestones 3 and 4)

### Setup of Data Pipeline within the Larger ML Pipeline/MLOps Platform

#### Data Validation and Verification
- Integrated validation step using ZenML and Qdrant APIs.
- Ensured data integrity by checking the presence, dimension consistency, and payload correctness of embeddings stored in Qdrant.
- Implemented automated checks to confirm successful data ingestion and proper storage.

#### Data Preprocessing and Feature Engineering
- Implemented a preprocessing pipeline step for:
  - Loading documents with `SimpleDirectoryReader`.
  - Cleaning and normalizing text using custom functions.
  - Chunking documents using `SentenceSplitter` (chunk size: 450 words, overlap: 50 words).
- Stored processed chunks into Qdrant with metadata, ensuring optimal retrieval.

#### Data Versioning
- Managed data versioning explicitly using DVC for both raw and preprocessed datasets.
- Tracked changes in dataset and embeddings, facilitating reproducibility of training and inference pipelines.

#### Setup Machine Learning Pipeline (ZenML)
- Defined clear ZenML pipeline steps for:
  - Data Preprocessing
  - Embedding generation and storage
  - Validation of embeddings
  - Querying and retrieving relevant data
- Configured pipeline to be executed reliably and reproducibly through ZenML's orchestration.

## 2.3 Pipeline Components Implementation

### Pipeline Structure and Components:

- **Data_preprocessing.py:**
  - Loads raw PDF documents.
  - Cleans and chunks text using `SentenceSplitter` and regular expressions.

- **index_storage.py:**
  - Embeds text chunks using `msmarco-distilbert-base-v4` and stores embeddings in Qdrant.

- **Validation.py:**
  - Verifies embedding storage correctness in Qdrant by validating vector existence and consistency.

- **Query_engine.py:**
  - Performs similarity search in Qdrant.
  - Retrieves relevant document chunks using embeddings.

- **Config.py:**
  - Centralized management of configurations, embedding model, and Qdrant credentials.

- **Main.py:**
  - Executes all pipeline components sequentially through ZenML pipeline orchestration.

## 2.4 Next Steps

- Extend pipeline to support real-time deployment through FastAPI.
- Incorporate monitoring and logging of model performance using MLflow.
- Enhance validation and verification steps for automatic anomaly detection and reporting.

---

This structured approach ensures reproducibility, scalability, and maintainability in the development and deployment of the AUIChat ML pipeline.

