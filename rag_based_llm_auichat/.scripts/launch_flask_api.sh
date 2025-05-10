#!/bin/bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI
# Create directory for PID files if it doesn't exist
mkdir -p /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/.pids
nohup python api.py > /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/flask_api.log 2>&1 &
echo $! > /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/.pids/flask_api.pid
