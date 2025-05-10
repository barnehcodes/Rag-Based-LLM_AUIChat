#!/bin/bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI/auichat
# Create directory for PID files if it doesn't exist
mkdir -p /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/.pids
# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi
nohup npm run dev > /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/react_ui.log 2>&1 &
echo $! > /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/.pids/react_ui.pid
