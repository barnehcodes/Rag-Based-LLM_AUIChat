# ML6 Milestone Progress: Model Testing, Evaluation, Monitoring & Continual Learning

This document tracks the progress of implementing ML6 features for the AUIChat RAG-based LLM ZenML pipeline.

## Overall Strategy:
Focus on enhancing the cloud deployment pipeline (`auichat_cloud_deployment_pipeline` in `src/main.py`) as it's better suited for online testing, managed monitoring, and robust CI/CD/CT. The local Seldon pipeline can still benefit from offline evaluation, bias audits, and explainability steps.

## Feature Breakdown, Feasibility, and Tools:

1.  **Model Evaluation (Unseen Data):**
    *   **Status:** Implemented (with Vertex AI integration)
    *   **Feasibility:** High.
    *   **Approach:** Integrated evaluation using Google Vertex AI and Cloud Functions. Tracks RAG-specific metrics over time and stores results in Cloud Storage. Implemented as ZenML step in cloud deployment pipeline.
    *   **Tools:**
        *   Google Cloud Vertex AI
        *   Google Cloud Functions
        *   Google Cloud Storage
        *   Google Cloud Scheduler
        *   ZenML (pipeline integration)
    *   **Implementation Details:**
        * Created `vertex_ai_evaluation.py` with a comprehensive implementation that:
            * Registers the existing Cloud Run RAG endpoint in Vertex AI Model Registry
            * Evaluates key RAG metrics: Context Precision, Context Recall, Faithfulness, and Answer Relevancy
            * Creates visualization dashboards for performance monitoring
            * Handles validation dataset with reference answers for automatic evaluation
        * Set up a complete centralized evaluation platform:
            * ZenML step integration (`vertex_ai_evaluation_step`)
            * Cloud Function for scheduled evaluations
            * Cloud Storage bucket for storing evaluation history
            * Cloud Scheduler job for daily automatic assessment
        * Test dataset of 5 AUI-specific questions created with reference answers
        * Includes both standalone evaluation script and pipeline integration
    *   **Next Steps:**
        * Implement alerts for metric degradation
        * Expand the test dataset for more comprehensive evaluation
        * Add evaluation tracking in Vertex AI Experiments
        * Integrate with MLflow for metrics visualization

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

1.  **Offline Evaluation:** Implement "Model evaluation using unseen data". (Currently blocked by API limits)
2.  **Basic Data Drift Detection:** Start with monitoring query embedding drift.
3.  **Robustness Testing:** Add a behavioral testing step with a few key test types.
