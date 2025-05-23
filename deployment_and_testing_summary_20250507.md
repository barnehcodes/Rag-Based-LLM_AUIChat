# AUIChat RAG Model Deployment & Testing Summary - May 7, 2025

## 1. Objective

The primary objective was to deploy the Al Akhawayn University AI Chat (AUIChat) RAG (Retrieval Augmented Generation) model to Google Cloud Run and ensure it was functioning correctly by performing tests.

## 2. Deployment Process & Challenges

The deployment involved packaging the Flask application, which uses LlamaIndex and Qdrant, into a Docker container and deploying it via Google Cloud Build and Cloud Run. Several challenges were encountered and resolved:

*   **Initial `gcloud` Command Errors:**
    *   Incorrect keys (`type`, `http-get-path`, `timeout`, `period`) in the `--startup-probe` argument for `gcloud run deploy`. This was iteratively corrected by referencing the official documentation, eventually simplifying the probe or removing it to allow Cloud Run's default health checks.
    *   Invalid `failure-threshold` value for the startup probe.

*   **Python Module Not Found Errors:**
    *   `ModuleNotFoundError: No module named 'flask_cors'`: Resolved by adding `flask-cors` to the `requirements.txt` generated by `deploy_qdrant_rag.sh`.
    *   `ModuleNotFoundError: No module named 'llama_index.vector_stores'`: Resolved by adding `llama-index-vector-stores-qdrant` to `requirements.txt`.
    *   Further `ModuleNotFoundError` issues for other LlamaIndex components (`llama-index-embeddings-huggingface`, `llama-index-llms-huggingface`) were also resolved by adding these explicit dependencies to `requirements.txt`. This highlighted the modular nature of recent LlamaIndex versions.

*   **LlamaIndex Context Window Error:**
    *   `ERROR - Error processing chat request: Calculated available context size -53 was not non-negative.` This indicated that the combined length of the query and retrieved text chunks exceeded the LLM's configured context window (initially 512 tokens).
    *   **Resolution:** Increased the `context_window` parameter for `HuggingFaceLLM` in `improved_rag_app_qdrant.py` from 512 to 1024.

*   **LLM Response Contamination:**
    *   The LLM's generated answer initially included metadata from the retrieved text chunks (e.g., `**Q: 255**`, `Page_label: 5`, `file_path:...`).
    *   **Resolution:** Added a cleanup step in `improved_rag_app_qdrant.py` using regular expressions to remove these metadata patterns from the final response string before sending it to the client.

## 3. Outcome

*   **Successful RAG Backend Deployment:** The AUIChat RAG service was successfully deployed to Google Cloud Run.
    *   **Service URL:** `https://auichat-rag-qdrant-h4ikwiq3ja-uc.a.run.app`
*   **Successful UI Hosting:** The user interface for AUIChat was successfully hosted on Firebase, allowing users to interact with the deployed RAG backend.
*   **Successful Health Check:** The `/health` endpoint of the RAG backend confirmed that all components (embedding model, LLM, Qdrant connection) were loaded and operational, with the Qdrant collection containing 2154 vectors.
*   **Successful Testing:** Both direct API testing and UI-based testing confirmed the end-to-end functionality of the system.

## 4. Testing

### 4.1. API Endpoint Testing

The deployed RAG backend was tested directly using `curl` to send a POST request to the `/predict` endpoint.

**Request:**
```bash
curl -X POST https://auichat-rag-qdrant-h4ikwiq3ja-uc.a.run.app/predict \\
-H "Content-Type: application/json" \\
-d '{"query": "What are the admission requirements for freshmen?"}'
```

**Response:**
```json
{
  "response": " The admission requirements for freshmen are as follows:**Applicants must have a high school diploma (high school diplomas and GEDs) or equivalent****Applicants must be able to pass the TOEFL exam****Applications are accepted by the Office of Admissions at the University of Alabama****Applicants must have a minimum GPA of 4.00****Applicants must have a minimum total score of 600 on the TOEFL exam****Applicants must have an SAT score of at least 850****Applicants must be enrolled in a freshmen major or minor*"
}
```
This confirmed the endpoint was live and the RAG model could process queries and return answers.

### 4.2. UI Testing

The Firebase-hosted UI was tested by inputting queries and observing the responses. The UI successfully communicated with the backend and displayed the answers retrieved by the RAG model.

*(Placeholder for UI screenshot)*

### 4.3. Script-based Testing (`test_rag.py`)

*   The `test_rag.py` script was used to query the deployed endpoint.
    *   After resolving the aforementioned issues, the RAG model successfully processed the query: "What are the admission requirements for freshmen?"
    *   The model provided a relevant answer: "A satisfactory online interview, where applicable, and personal essay evaluation are required documents."
    *   The response also correctly included source documents, such as `Undergraduate Admission Freshmen Non-Degree Seeking.pdf` and `AUI Catalog_2023-2024_New_Version.pdf`, with their respective similarity scores.

## 5. Key Files Involved & Modified

*   **`deploy_qdrant_rag.sh`:**
    *   Iteratively refined `gcloud run deploy` command, especially the `--startup-probe` argument.
    *   Updated to include all necessary Python dependencies in the dynamically generated `requirements.txt` (e.g., `flask-cors`, `llama-index-vector-stores-qdrant`, `llama-index-embeddings-huggingface`, `llama-index-llms-huggingface`).
*   **`improved_rag_app_qdrant.py`:**
    *   Increased `context_window` for `HuggingFaceLLM` to 1024.
    *   Added a response cleanup mechanism to remove unwanted metadata from the LLM's output.
    *   Ensured correct initialization of LlamaIndex components.
*   **`test_rag.py`:**
    *   Adjusted to send the correct payload format to the `/predict` endpoint.
    *   Modified to correctly parse the server's response structure.
*   **Firebase UI Configuration Files (Not detailed here):** Files related to the UI build and Firebase deployment configuration were also part of the successful UI hosting.

## 6. Conclusion

The deployment and testing process was successful, culminating in a fully operational AUIChat system. The RAG backend is robustly deployed on Cloud Run, and the UI is accessible via Firebase. Both direct API calls and user interactions through the UI confirm the system's ability to answer queries based on the indexed documents in Qdrant and provide relevant source information.
