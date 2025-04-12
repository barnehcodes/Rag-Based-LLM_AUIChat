#!/bin/bash
# run_integrated_auichat.sh - Script to launch the AUIChat UI with RAG backend

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Starting AUIChat with RAG Integration${NC}"
echo -e "${BLUE}==================================================${NC}"

# Set paths
ROOT_DIR="$(dirname "$(readlink -f "$0")")"
UI_DIR="$ROOT_DIR/src/UI/auichat"
API_DIR="$ROOT_DIR/src/UI"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to kill processes when the script exits
cleanup() {
  echo -e "\n${YELLOW}Shutting down services...${NC}"
  if [ ! -z "$API_PID" ]; then
    echo "Stopping API server (PID: $API_PID)"
    kill $API_PID 2>/dev/null
  fi
  if [ ! -z "$UI_PID" ]; then
    echo "Stopping UI server (PID: $UI_PID)"
    kill $UI_PID 2>/dev/null
  fi
  echo -e "${GREEN}All services stopped.${NC}"
  exit 0
}

# Set trap for cleanup when script is terminated
trap cleanup SIGINT SIGTERM

# Step 1: Start the Flask API Server
echo -e "\n${YELLOW}Starting RAG API Server...${NC}"
cd "$API_DIR"
# Check if Python is available
if command_exists python3; then
  python3 api.py &
  API_PID=$!
elif command_exists python; then
  python api.py &
  API_PID=$!
else
  echo -e "${RED}Error: Python not found.${NC}"
  exit 1
fi
echo -e "${GREEN}✓ API Server started with PID: $API_PID${NC}"
echo -e "  API running at: http://localhost:5000"

# Wait a bit for the API to start
sleep 2

# Step 2: Start the React Development Server
echo -e "\n${YELLOW}Starting React Frontend...${NC}"
cd "$UI_DIR"

# Check if npm is available
if command_exists npm; then
  echo "Using npm to start frontend"
  npm install
  npm run dev &
  UI_PID=$!
# Check if yarn is available
elif command_exists yarn; then
  echo "Using yarn to start frontend"
  yarn install
  yarn dev &
  UI_PID=$!
# Check if bun is available
elif command_exists bun; then
  echo "Using bun to start frontend"
  bun install
  bun run dev &
  UI_PID=$!
else
  echo -e "${RED}Error: No package manager (npm, yarn, or bun) found.${NC}"
  # Kill the API server
  kill $API_PID
  exit 1
fi

echo -e "${GREEN}✓ React Frontend started with PID: $UI_PID${NC}"
echo -e "  Frontend running at: http://localhost:5173"

echo -e "\n${BLUE}==================================================${NC}"
echo -e "${GREEN}AUIChat is now running!${NC}"
echo -e "${YELLOW}Access the application at: ${NC}http://localhost:5173"
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${BLUE}==================================================${NC}"

# Keep the script running
wait