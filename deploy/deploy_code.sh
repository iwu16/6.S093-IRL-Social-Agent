#!/bin/bash
# Deploy code to GCP VM via IAP
# Syncs project files and sets up the Python environment

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-iap-sidequest}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-irl-social-agent}"
REMOTE_DIR="/home/$(whoami)/irl-social-agent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory (deploy folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}=== Deploying Code to VM ===${NC}"
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE_NAME"
echo "Local: $PROJECT_DIR"
echo "Remote: $REMOTE_DIR"
echo ""

# Check if .env file exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Files to sync (exclude .git, .venv, __pycache__, etc.)
echo -e "${YELLOW}Syncing files to VM...${NC}"

# Create remote directory
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    --command="mkdir -p $REMOTE_DIR"

# Copy project files using gcloud scp
gcloud compute scp \
    --zone="$ZONE" \
    --tunnel-through-iap \
    --recurse \
    "$PROJECT_DIR"/*.py \
    "$PROJECT_DIR"/requirements.txt \
    "$PROJECT_DIR"/pyproject.toml \
    "$PROJECT_DIR"/schema.sql \
    "$PROJECT_DIR"/.env \
    "$INSTANCE_NAME:$REMOTE_DIR/"

echo -e "${GREEN}Files synced!${NC}"
echo ""

# Set up Python environment on VM
echo -e "${YELLOW}Setting up Python environment on VM...${NC}"
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    --command="cd $REMOTE_DIR && \
        export PATH=\"\$HOME/.local/bin:\$PATH\" && \
        if ! command -v uv &> /dev/null; then
            echo 'Installing UV...'
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH=\"\$HOME/.local/bin:\$PATH\"
        fi && \
        if [ ! -d .venv ]; then
            echo 'Creating virtual environment...'
            uv venv
        fi && \
        echo 'Installing dependencies...' && \
        source .venv/bin/activate && \
        uv pip install -r requirements.txt && \
        echo 'Initializing database...' && \
        python3 database.py && \
        echo 'Setup complete!'"

echo ""
echo -e "${GREEN}=== Code Deployed Successfully ===${NC}"
echo ""
echo "To run the agent on the VM:"
echo -e "${YELLOW}  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --tunnel-through-iap${NC}"
echo "  Then: cd $REMOTE_DIR && source .venv/bin/activate && python main.py --help"
