#!/bin/bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI
# Create directory for PID files if it doesn't exist
mkdir -p /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/.pids
nohup python cors_proxy.py > /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/cors_proxy.log 2>&1 &
echo $! > /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/.pids/cors_proxy.pid
