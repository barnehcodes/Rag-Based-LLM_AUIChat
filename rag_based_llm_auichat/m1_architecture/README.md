# Milestone 1: ML System Architecture

This milestone covers the architectural design of the AUIChat RAG-based system.

## Contents

- `mindMap.html`: Interactive visualization of the AUIChat system architecture
- Architecture diagrams and documentation

## Architecture Overview

The AUIChat system follows a RAG (Retrieval-Augmented Generation) architecture with the following key components:

1. **Data Pipeline**
   - Document Processing (PDF parsing, text extraction)
   - Preprocessing (cleaning, metadata tagging)
   - Chunking Strategies (size optimization, overlap settings)

2. **Vector Database**
   - Qdrant Integration for vector storage
   - Validation Workflows

3. **LLM Integration**
   - Mistral-7B-Instruct as the base model
   - Prompt Engineering

## Resources

- Use the mindMap.html to interactively explore the system architecture
- See the main project README for overall system documentation