# Model Serving

This directory contains components related to serving the AUIChat RAG model.

## Contents

- **test_vertex_endpoint.py**: Script to test the deployed Vertex AI endpoint
- **README.md**: This documentation file

## Key Components

The model serving layer includes:

1. **Vertex AI Integration**: Deployment configuration and endpoints for model serving
2. **Inference API**: Structured API for making predictions with the model
3. **Response Formatting**: Processing raw model outputs into structured responses
4. **Monitoring**: Logging and alerting for model serving health

## Using the Vertex AI Endpoint

After deploying the model to Vertex AI, you can test it using:

```bash
python test_vertex_endpoint.py --endpoint-id <YOUR_ENDPOINT_ID>
```

## Next Steps

- Implement additional model serving backends (Seldon, TorchServe)
- Add response caching for frequently asked questions
- Implement rate limiting and authentication