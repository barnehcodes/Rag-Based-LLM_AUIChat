# AUIChat CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline set up for the AUIChat project using GitHub Actions.

## Overview

The CI/CD pipeline automates the following processes:
- Code linting and testing
- Building Docker containers for the RAG application and UI
- Deploying to either local Kubernetes (with Seldon Core) or Google Cloud Run
- Running ZenML pipelines for full deployment workflows

## GitHub Actions Workflow

The workflow is defined in `.github/workflows/auichat-cicd.yml` and includes the following jobs:

1. **test**: Lints the code with flake8 and runs unit tests
2. **build-rag-app**: Builds a Docker image for the RAG backend service
3. **build-ui**: Builds a Docker image for the React UI
4. **deploy-cloud**: Deploys both containers to Google Cloud Run (for production)
5. **deploy-zenml-pipeline**: Runs ZenML pipelines (either local or cloud)
6. **notify**: Sends notifications about deployment results

## Triggering the Pipeline

The pipeline can be triggered in multiple ways:

1. **Automatic triggers**:
   - Push to `main` branch: Runs tests, builds, and deploys to Cloud Run
   - Push to `dev` branch: Runs tests and builds but doesn't deploy
   - Pull request to `main`: Runs tests only

2. **Manual triggers** via GitHub Actions UI:
   - Select "Actions" in your GitHub repo
   - Choose "AUIChat CI/CD Pipeline"
   - Click "Run workflow"
   - Select one of these deployment types:
     - `test`: Run tests only
     - `local`: Deploy to local Kubernetes with Seldon
     - `cloud`: Deploy to Google Cloud Run

## Required Secrets

The workflow requires the following secrets to be configured in your GitHub repository:

- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_SA_KEY`: Service account key JSON for GCP authentication (with permissions for GCR and Cloud Run)
- `QDRANT_URL`: URL for your Qdrant vector database (optional if using default)
- `QDRANT_API_KEY`: API key for your Qdrant database (optional if using default)
- `SLACK_WEBHOOK_URL`: URL for Slack notifications (optional)

To add these secrets:
1. Go to your GitHub repository
2. Click Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Add each secret with its appropriate value

## Deployment Artifacts

After a successful deployment to Cloud Run, the workflow creates and uploads a `cloudrun_deployment_info.json` file as an artifact. This file contains:

- RAG and UI service URLs
- Docker image references
- Project and region information
- Timestamp of deployment

You can find these artifacts by:
1. Going to the Actions tab in your GitHub repository
2. Selecting the completed workflow run
3. Scrolling down to the "Artifacts" section
4. Downloading the "deployment-info" file

## Local vs. Cloud Deployments

The workflow supports two main deployment targets:

### Local Deployment (ZenML with Seldon Core)
- Uses a local Kubernetes cluster with Seldon Core
- Deployed using your ZenML local pipeline
- Good for development and testing
- Note: Running this in GitHub Actions requires a self-hosted runner with Kubernetes

### Cloud Deployment (Google Cloud Run)
- Deploys containers directly to Cloud Run
- Optimized for production use
- Fully managed and auto-scaling
- All configuration is handled via environment variables

## Troubleshooting

If deployments fail, check the following:

1. **Authentication issues**:
   - Verify GCP service account key has the necessary permissions
   - Check that Qdrant credentials are correct

2. **Build failures**:
   - Look at the build logs for dependency or syntax errors
   - Ensure the Docker build steps can locate all required files

3. **Deployment issues**:
   - Check Cloud Run logs for runtime errors
   - Verify ZenML is properly configured with all required components

## Extending the Pipeline

To extend the pipeline for additional functionality:

1. **Add testing tools**: Modify the `test` job to include additional tools
2. **Add deployment targets**: Create new jobs similar to `deploy-cloud`
3. **Add artifact publishing**: Enhance the notification job to store artifacts in additional locations

## Continuous Monitoring

While not included in this workflow, consider adding:
- Integration with monitoring tools
- Automated testing of deployed services
- Performance benchmark comparisons between deployments