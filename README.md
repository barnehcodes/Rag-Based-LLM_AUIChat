# **ML Pipeline Development - Milestone 3 Report**

## **1. Introduction**

This report outlines the progress made in **Milestone 3: Data Acquisition and Preparation** as part of the larger **ML Pipeline Development** for AUIChat, a RAG-based chatbot. This milestone involved structuring the data pipeline, implementing data validation, preprocessing, embedding generation, and storage. Additionally, the pipeline was integrated into ZenML to enhance reproducibility and scalability.

## **2. Ensuring ML Pipeline Reproducibilit** 



### ✅ **2.1 Project Structure Definition and Modulari**

- **Where?** The project was modularized into separate components under the `pipeline/` directory:
  - `Data_preprocessing.py` → Data loading, cleaning, chunking
  - `index_storage.py` → Embedding generation & vector storage
  - `query_engine.py` → Query processing and retrieval
  - `validation.py` → Data validation
  - `config.py` → Centralized configurations
  - `main.py` → Pipeline execution
- **Status:** ✅ **Done**

### ✅ **2.2 Code Versioning** 

- **Where?** Git was used for version control.
- **How?** Commits were structured for each milestone.
- **Status:** ✅ **Done**

### ✅ **2.3 Data Versioning** 

- **Where?** Implemented using **DVC (Data Version Control)**.
- **How?**
  - Dataset stored under `pipeline/resources/`
  - Tracked using DVC to allow reproducible dataset versions.
  - Issue with Git tracking (`git rm --cached pipeline/resources` resolved).
- **Status:** ✅ **Done**

### ✅ **2.4 Experiment Tracking and Model Versioning** 

- **Where?**
  - **ZenML** tracks pipeline execution.
  - **MLflow** (optional) for additional logging and model versioning.
- **Status:** ✅ **Done**

### ✅ **2.5 Setting Up a Meta Store for Metadata**

- **Where?**
  - **Qdrant** stores metadata (file names, chunk IDs, timestamps, versions).
  - **ZenML** tracks metadata at each pipeline step.
- **Status:** ✅ **Done**

### ✅ **2.6 Setting Up the ML Pipeline Under an MLOps Platform** 

- **Where?** Integrated with **ZenML** to orchestrate data processing, indexing, and querying.
- **How?**
  - `@pipeline` decorator in `pipeline.py`.
  - Steps (`@step`) in `Data_preprocessing.py`, `index_storage.py`, `query_engine.py`, `validation.py`.
- **Status:** ✅ **Done**

---

## **3. Pipeline Components** 

![Pipeline Diagram](Rag-Based-LLM_AUIChat/Pipeline Structure and Components.png)

### ✅ **3.1 Setup of Data Pipeline Within the ML Pipeline / MLOps Platform**

#### ✅ **3.1.1 Data Validation and Verification** 

- **Where?** Implemented in `validation.py`.
- **How?**
  - Checks Qdrant storage integrity.
  - Validates embedding existence & metadata consistency.
  - Uses `validate_qdrant_storage()`.
- **Status:** ✅ **Done**

#### ✅ **3.1.2 Preprocessing and Feature Engineering** 

- **Where?** Implemented in `Data_preprocessing.py`.
- **How?**
  - **Cleaning**: Converts text to lowercase, removes special characters.
  - **Chunking**: Uses `SentenceSplitter` (chunk size = 450, overlap = 50).
  - **Feature Store Integration**: Qdrant stores preprocessed embeddings.
- **Status:** ✅ **Done**

### ✅ **3.2 Integration of Model Training and Offline Evaluation into the ML Pipeline / MLOps Platform** 

- **Where?** Not applicable in this milestone since **AUIChat uses retrieval-based methods instead of model training.**
- **Future Work:** Model fine-tuning for ranking retrieved documents.
- **Status:** ❌ **Not Required in this Milestone**

### ✅ **3.3 Development of Model Behavioral Tests** 

- **Where?** Implemented in `validation.py`.
- **How?**
  - Retrieval consistency tests.
  - Ensures Qdrant stores valid embeddings & metadata.
- **Status:** ✅ **Done**

---

## **4. Summary of Achievements**

 **Fully modular pipeline setup under ZenML.** **Data ingestion, validation, and storage completed.**  **Qdrant used as a vector store + metadata store.**  **Data versioning with DVC.**  **Experiment tracking via ZenML.**  **Retrieval pipeline structured for fast response times.**

##

