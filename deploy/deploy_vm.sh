#!/bin/bash
# Deploy IRL Social Agent to Google Cloud VM
# Uses IAP for secure SSH access

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-iap-sidequest}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-irl-social-agent}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== IRL Social Agent VM Deployment ===${NC}"
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine Type: $MACHINE_TYPE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not installed${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}Setting project...${NC}"
gcloud config set project "$PROJECT_ID"

# Check if instance already exists
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
    echo -e "${YELLOW}Instance $INSTANCE_NAME already exists${NC}"
    read -p "Do you want to delete and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deleting existing instance...${NC}"
        gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet
    else
        echo "Keeping existing instance. Use 'deploy_code.sh' to update code."
        exit 0
    fi
fi

# Create the VM instance
echo -e "${YELLOW}Creating VM instance...${NC}"
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-balanced \
    --tags=iap-ssh \
    --metadata-from-file=startup-script="$SCRIPT_DIR/startup-script.sh"

echo -e "${GREEN}VM created successfully!${NC}"
echo ""

# Configure IAP firewall rule
echo -e "${YELLOW}Configuring IAP firewall rule...${NC}"
if ! gcloud compute firewall-rules describe allow-iap-ssh --project="$PROJECT_ID" &> /dev/null; then
    gcloud compute firewall-rules create allow-iap-ssh \
        --project="$PROJECT_ID" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp:22 \
        --source-ranges=35.235.240.0/20 \
        --target-tags=iap-ssh \
        --description="Allow SSH via IAP"
    echo -e "${GREEN}Firewall rule created${NC}"
else
    echo "Firewall rule already exists"
fi

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "To connect to your VM via IAP SSH:"
echo -e "${YELLOW}  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --tunnel-through-iap${NC}"
echo ""
echo "To deploy code to the VM:"
echo -e "${YELLOW}  ./deploy/deploy_code.sh${NC}"
