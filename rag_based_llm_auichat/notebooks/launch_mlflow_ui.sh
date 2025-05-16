#!/bin/bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/notebooks

# Activate the same Python environment as the current process (if in a virtualenv)
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Activating virtualenv: $VIRTUAL_ENV"
    source "$VIRTUAL_ENV/bin/activate"
fi

# Install required dependencies
pip install "jinja2<3.1" flask "werkzeug<3.0.0" > /dev/null 2>&1

# Clean up any previous log/pid
rm -f mlflow_ui.log mlflow_ui.pid

# Start MLflow UI
nohup mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5004 > mlflow_ui.log 2>&1 &
echo $! > mlflow_ui.pid
echo "MLflow UI started on port 5004. PID saved to mlflow_ui.pid."
echo "Dashboard available at: http://localhost:5004"
echo "Log file: mlflow_ui.log"

# Verify startup - wait a moment for process to start
sleep 2
if ! ps -p $(cat mlflow_ui.pid 2>/dev/null) > /dev/null 2>&1; then
    echo "ERROR: MLflow UI process failed to start or died immediately"
    echo "Check log file for details: mlflow_ui.log"
    exit 1
fi
