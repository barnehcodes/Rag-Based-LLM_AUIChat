# ML6 Milestone Progress: Model Testing, Evaluation, Monitoring & Continual Learning

This document tracks the progress of implementing ML6 features for the AUIChat RAG-based LLM ZenML pipeline.

## Overall Strategy:
Focus on enhancing the cloud deployment pipeline (`auichat_cloud_deployment_pipeline` in `src/main.py`) as it's better suited for online testing, managed monitoring, and robust CI/CD/CT. The local Seldon pipeline can still benefit from offline evaluation, bias audits, and explainability steps.

## Feature Breakdown, Feasibility, and Tools:

1.  **Model Evaluation (Unseen Data):**
    *   **Status:** Not Started
    *   **Feasibility:** High.
    *   **Approach:** Add a ZenML step after model deployment (or before promotion in CT). Curate an unseen dataset. Calculate RAG-specific metrics (Faithfulness, Answer Relevancy, Context Precision/Recall). Log metrics to MLflow.
    *   **Tools:**
        *   Python scripts
        *   RAGAs (library)
        *   MLflow

2.  **Online Testing (A/B Testing, Bandit):**
    *   **Status:** Not Started
    *   **Feasibility:** Medium (A/B testing is more straightforward).
    *   **Approach (A/B Testing):** Deploy two versions of the model on Cloud Run. Use Cloud Run's traffic splitting. Collect performance metrics for both. ZenML orchestrates challenger deployment.
    *   **Tools:**
        *   Google Cloud Run (traffic splitting)
        *   ZenML

3.  **Audit Model for Bias:**
    *   **Status:** Not Started
    *   **Feasibility:** Medium. Requires careful dataset creation.
    *   **Approach:** ZenML step in both local and cloud. Develop a dataset with queries to reveal biases. Analyze responses.
    *   **Tools:**
        *   Custom Python scripts for test suites.

4.  **Behavioral or Adversarial Testing:**
    *   **Status:** Not Started
    *   **Feasibility:** Medium.
    *   **Approach:** ZenML step. Create test cases with perturbations (typos, paraphrases, etc.). Evaluate model consistency and robustness.
    *   **Tools:**
        *   Python libraries like `nlpaug` or custom scripts.
        *   Potentially LangTest, CheckList.

5.  **Model Explainability and Interpretability (for RAG):**
    *   **Status:** Partially Addressed (RAG inherently shows retrieved context).
    *   **Feasibility:** High for RAG's context.
    *   **Approach:** Enhance logging/output to clearly show retrieved document chunks used for answers.
    *   **Tools:**
        *   Python (modify model serving code or add ZenML step).

6.  **Resource Monitoring:**
    *   **Status:** Partially Addressed (Cloud provider defaults).
    *   **Feasibility:** High (for cloud).
    *   **Approach:** Leverage Google Cloud Monitoring for Cloud Run services.
    *   **Tools:**
        *   Google Cloud Monitoring.

7.  **Data Drift Monitoring:**
    *   **Status:** Not Started
    *   **Feasibility:** Medium.
    *   **Approach:** ZenML step. Monitor query distribution (embedding drift) and document drift against a baseline.
    *   **Tools:**
        *   Evidently AI, NannyML, or WhyLabs (integrated with ZenML).
        *   Custom Python scripts for statistical tests on embeddings.

8.  **Model Performance Monitoring:**
    *   **Status:** Not Started
    *   **Feasibility:** Medium. Requires feedback loop or proxy metrics.
    *   **Approach:** Track RAG-specific metrics over time using production data/feedback. Monitor proxy metrics. Set alerts.
    *   **Tools:**
        *   MLflow (for logging production metrics).
        *   Evidently AI, NannyML, WhyLabs.
        *   Cloud Monitoring/Dashboards.

9.  **Continual Learning Pipeline Component (CT/CD):**
    *   **Status:** Partially Addressed (CI/CD for deployment exists).
    *   **Feasibility:** High for orchestration.
    *   **Approach:** Automate the `auichat_cloud_deployment_pipeline` to run based on triggers (schedule, data drift, model degradation, new code/data).
    *   **Tools:**
        *   ZenML (orchestration).
        *   GitHub Actions (triggers, CI/CD).
        *   MLflow (model versioning/registry).

## Easiest and Most Effective First Steps:

1.  **Offline Evaluation:** Implement "Model evaluation using unseen data".
2.  **Basic Data Drift Detection:** Start with monitoring query embedding drift.
3.  **Robustness Testing:** Add a behavioral testing step with a few key test types.
