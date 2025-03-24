from zenml import step
import mlflow

@step
def evaluate_response(query: str, llm_response: list):
    """
    Evaluate a response from the LLM and log it in MLflow.
    """
    if not llm_response or not isinstance(llm_response, list):
        print("⚠ No valid response to evaluate.")
        return

    top_response = llm_response[0]["text"] if "text" in llm_response[0] else "No text found"
    
    print(f"🧠 Evaluating LLM Response:\n{top_response}\n")

    with mlflow.start_run(run_name="response_evaluation"):
        mlflow.log_param("query", query)
        mlflow.log_metric("response_length", len(top_response))
        mlflow.log_artifact("query_response.txt")

        # Save response to file
        with open("query_response.txt", "w") as f:
            f.write(f"Query: {query}\n\nResponse:\n{top_response}")
