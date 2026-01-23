#!/bin/bash
# VM Startup Script - Installs Python, UV, and dependencies

apt-get update
apt-get install -y python3 python3-pip python3-venv curl git

# Install UV for all users
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile.d/uv.sh

echo "VM setup complete!"
