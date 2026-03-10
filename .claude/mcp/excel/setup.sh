#!/bin/bash
# Setup script for excel MCP server
# Creates virtual environment and installs openpyxl + mcp + pyprojroot
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up excel MCP server..."

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Remove old venv if exists
if [[ -d ".venv" ]]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi

# Create fresh virtual environment
echo "Creating virtual environment..."
uv venv .venv --python python3

# Install dependencies
echo "Installing dependencies..."
uv pip install --python .venv/bin/python "mcp>=1.0.0" "openpyxl>=3.1.0"

echo ""
echo "Setup complete!"
echo "To test: .venv/bin/python mcp_server.py"
