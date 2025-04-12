# AUIChat - RAG-Based LLM Chat Interface

A modern React-based chat interface for the RAG-Based LLM AUIChat application, providing AI-powered information about Al Akhawayn University.

## Overview

AUIChat is a Retrieval-Augmented Generation (RAG) powered conversational assistant that provides accurate and contextual information about Al Akhawayn University policies, programs, and services. The application uses a React frontend with a Flask API backend connected to a vector database.

## How to Run

### Prerequisites

- Node.js (v16+)
- npm or yarn
- Python 3.10+
- Flask and other Python dependencies

### Step 1: Start the Flask API Server (Backend)

```bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI
python api.py
```

This starts the main Flask API server on port 5000. Note that initialization may take up to a minute as it loads machine learning models and connects to the vector database.

### Step 2: Start the CORS Proxy

Due to CORS restrictions in browsers, a proxy is needed to handle cross-origin requests between the frontend and backend:

```bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI
python cors_proxy.py
```

This starts a CORS proxy on port 5001 that facilitates communication between the React UI and Flask API.

### Step 3: Start the React UI

```bash
cd /home/barneh/Rag-Based-LLM_AUIChat/rag_based_llm_auichat/src/UI/auichat
npm install  # Only needed the first time or when dependencies change
npm run dev
```

This starts the Vite development server, typically on port 5173.

### Step 4: Access the Application

Open your web browser and navigate to:
```
http://localhost:5173
```

## API Documentation

The AUIChat application uses a RESTful API built with Flask to handle chat interactions:

### Endpoints

#### POST `/api/chat`

Processes user queries and returns AI-generated responses.

**Request Body:**
```json
{
  "message": "What are the requirements for the PiP program?"
}
```

**Response Body:**
```json
{
  "response": "The PiP (Partnership in Programming) program has the following requirements: ...",
  "metrics": {
    "inferenceTime": 2500.45
  }
}
```

#### GET `/api/health`

Simple health check endpoint to verify the API is running.

**Response Body:**
```json
{
  "status": "healthy",
  "service": "auichat-rag-api"
}
```

## Architecture

The application consists of three main components:

1. **React Frontend**: Modern, responsive UI built with React, Material-UI, and Anime.js for animations
2. **CORS Proxy**: Flask-based proxy server that handles cross-origin requests
3. **Flask API Backend**: Handles chat requests, connects to the RAG pipeline

### RAG Pipeline

The backend uses a Retrieval-Augmented Generation (RAG) pipeline that:

1. Converts user questions into vector embeddings
2. Retrieves relevant information from a Qdrant vector database
3. Uses Mistral-7B-Instruct LLM to generate contextually accurate responses

## Features

- **Dynamic Chat Interface**: Real-time chat with smooth animations
- **Suggested Queries**: Common questions provided as chips for easy access
- **Performance Metrics**: Display of inference time to show processing performance
- **Response Animation**: Fluid animations for chat messages
- **Mobile Responsive**: Fully responsive design for all device sizes

## Troubleshooting

- **API Connection Issues**: Ensure both the Flask API (port 5000) and CORS proxy (port 5001) are running
- **Long Initial Response**: The first query may take longer as the LLM model warms up
- **CORS Errors**: If you see CORS errors in the console, verify the CORS proxy is running
- **"NetworkError when attempting to fetch resource"**: Ensure the API server has fully initialized (can take up to a minute after starting)

## Development

### Project Structure

- `src/components/ChatInterface`: Main chat UI component
- `src/components/Theme`: Theme configuration
- `src/components/DynamicBackground`: Background animation effects
- `src/components/TechShowcase`: Technology showcase component
- `src/components/Header`: Application header

### Building for Production

```bash
npm run build
```

This generates optimized production files in the `dist` directory that can be served by any static file server.
