#!/bin/bash
# Deploy CORS proxy service for AUIChat
# This script deploys a CORS proxy service that sits between your UI and RAG endpoint

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID="deft-waters-458118-a3"
REGION="us-central1"
RAG_ENDPOINT="https://auichat-rag-qdrant-h4ikwiq3ja-uc.a.run.app"
UI_ENDPOINT="https://auichat-ui-h4ikwiq3ja-uc.a.run.app"

# Display help message
function show_help {
    echo -e "${BLUE}AUIChat CORS Proxy Deployment Script${NC}"
    echo "Usage: ./deploy_cors_proxy.sh [options]"
    echo ""
    echo "Options:"
    echo "  -p, --project     GCP project ID (default: deft-waters-458118-a3)"
    echo "  -r, --region      GCP region (default: us-central1)"
    echo "  -e, --rag-endpoint RAG endpoint URL (default: from cloudrun_qdrant_info.json)"
    echo "  -u, --ui-endpoint UI endpoint URL (default: from cloudrun_ui_info.json)"
    echo "  -h, --help        Show this help message"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--project)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -e|--rag-endpoint)
            RAG_ENDPOINT="$2"
            shift 2
            ;;
        -u|--ui-endpoint)
            UI_ENDPOINT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in the correct directory
PROJECT_ROOT=$(dirname $(realpath $0))
if [[ ! -f "$PROJECT_ROOT/cors_proxy_service.py" ]]; then
    echo -e "${RED}Error: cors_proxy_service.py not found${NC}"
    exit 1
fi

# Try to get endpoints from config files if not provided
if [[ -z "$RAG_ENDPOINT" && -f "$PROJECT_ROOT/cloudrun_qdrant_info.json" ]]; then
    RAG_ENDPOINT=$(grep -o '"service_url": *"[^"]*"' "$PROJECT_ROOT/cloudrun_qdrant_info.json" | awk -F'"' '{print $4}')
    echo -e "${BLUE}Using RAG endpoint from cloudrun_qdrant_info.json: $RAG_ENDPOINT${NC}"
fi

if [[ -z "$UI_ENDPOINT" && -f "$PROJECT_ROOT/cloudrun_ui_info.json" ]]; then
    UI_ENDPOINT=$(grep -o '"service_url": *"[^"]*"' "$PROJECT_ROOT/cloudrun_ui_info.json" | awk -F'"' '{print $4}')
    echo -e "${BLUE}Using UI endpoint from cloudrun_ui_info.json: $UI_ENDPOINT${NC}"
fi

echo -e "${BLUE}Deploying CORS proxy service with the following configuration:${NC}"
echo -e "  Project ID: ${GREEN}$PROJECT_ID${NC}"
echo -e "  Region: ${GREEN}$REGION${NC}"
echo -e "  RAG Endpoint: ${GREEN}$RAG_ENDPOINT${NC}"
echo -e "  UI Endpoint: ${GREEN}$UI_ENDPOINT${NC}"

# Build and deploy the CORS proxy service
echo -e "${BLUE}Building and deploying CORS proxy service...${NC}"

# Build the Docker image
IMAGE_NAME="gcr.io/$PROJECT_ID/auichat-cors-proxy:latest"
echo -e "${BLUE}Building Docker image: $IMAGE_NAME${NC}"
docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile.cors-proxy" "$PROJECT_ROOT"

# Push the image to Google Container Registry
echo -e "${BLUE}Pushing Docker image to GCR...${NC}"
gcloud auth configure-docker -q
docker push "$IMAGE_NAME"

# Deploy to Cloud Run
echo -e "${BLUE}Deploying to Cloud Run...${NC}"
gcloud run deploy auichat-cors-proxy \
    --image "$IMAGE_NAME" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --allow-unauthenticated \
    --set-env-vars="RAG_ENDPOINT=$RAG_ENDPOINT" \
    --quiet

# Get the service URL
SERVICE_URL=$(gcloud run services describe auichat-cors-proxy \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

# Save deployment info
cat > "$PROJECT_ROOT/cors_proxy_info.json" << EOF
{
  "service_name": "auichat-cors-proxy",
  "service_url": "$SERVICE_URL",
  "project_id": "$PROJECT_ID",
  "region": "$REGION",
  "image_name": "$IMAGE_NAME",
  "rag_endpoint": "$RAG_ENDPOINT",
  "ui_endpoint": "$UI_ENDPOINT",
  "deployment_date": "$(date -u "+%Y-%m-%d %H:%M:%S UTC")"
}
EOF

echo -e "${GREEN}CORS Proxy service deployed successfully!${NC}"
echo -e "${GREEN}Service URL: $SERVICE_URL${NC}"

# Now let's update the UI to use the CORS proxy instead of directly calling the RAG endpoint
echo -e "${BLUE}Now we need to update the UI configuration to use the CORS proxy...${NC}"

# Create a script to update the UI configuration
cat > "$PROJECT_ROOT/update_ui_config.sh" << EOF
#!/bin/bash
# Update UI configuration to use the CORS proxy

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID="$PROJECT_ID"
REGION="$REGION"
CORS_PROXY_URL="$SERVICE_URL"
UI_ENDPOINT="$UI_ENDPOINT"

echo -e "${BLUE}Updating UI configuration to use CORS proxy at $CORS_PROXY_URL${NC}"

# Create the environment variables file for the UI
cat > .env << EOF2
VITE_API_URL=$CORS_PROXY_URL
VITE_RAG_ENDPOINT=$CORS_PROXY_URL
# Environment variables for AUIChat UI
# Generated by update_ui_config.sh on \$(date)
EOF2

echo -e "${GREEN}Created .env file with CORS proxy configuration${NC}"

# Rebuild and redeploy the UI
echo -e "${BLUE}Building UI...${NC}"
npm run build

echo -e "${BLUE}Deploying updated UI to Cloud Run...${NC}"
# The simplified deploy command (replace with your actual deployment process)
gcloud run deploy \$(basename $UI_ENDPOINT | cut -d "." -f 1) \\
    --source . \\
    --region $REGION \\
    --project $PROJECT_ID \\
    --allow-unauthenticated

echo -e "${GREEN}UI redeployed successfully!${NC}"
echo -e "${GREEN}Your updated UI is available at: $UI_ENDPOINT${NC}"
EOF

chmod +x "$PROJECT_ROOT/update_ui_config.sh"

echo -e "${BLUE}======== Next Steps ========${NC}"
echo -e "1. To update your UI to use the CORS proxy, run the following commands:"
echo -e "   ${YELLOW}cd $PROJECT_ROOT/rag_based_llm_auichat/src/UI/auichat${NC}"
echo -e "   ${YELLOW}$PROJECT_ROOT/update_ui_config.sh${NC}"
echo -e ""
echo -e "2. Or you can manually update your UI environment variables:"
echo -e "   ${YELLOW}VITE_API_URL=$SERVICE_URL${NC}"
echo -e "   ${YELLOW}VITE_RAG_ENDPOINT=$SERVICE_URL${NC}"
echo -e ""
echo -e "3. After updating the environment variables, rebuild and redeploy your UI."
echo -e ""
echo -e "${GREEN}Your CORS proxy is now available at: $SERVICE_URL${NC}"