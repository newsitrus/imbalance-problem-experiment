#!/bin/bash
# Setup script for mineru-parser MCP server
# Installs MinerU (CPU pipeline backend) + mcp into an isolated .venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up mineru-parser MCP server..."

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

# Create fresh virtual environment (Python 3.10+)
echo "Creating virtual environment..."
uv venv .venv --python python3

# Install mcp
echo "Installing mcp..."
uv pip install --python .venv/bin/python "mcp>=1.0.0"

# Install mineru — try [pipeline] extra first (CPU-only deps), fall back to plain
echo "Installing mineru (pipeline backend, CPU-only)..."
uv pip install --python .venv/bin/python "mineru[pipeline]" || {
    echo "[pipeline] extra not available, installing plain mineru..."
    uv pip install --python .venv/bin/python mineru
}

echo ""
echo "Setup complete!"
echo ""
echo "IMPORTANT: The first parse call will download MinerU model weights (~2-5GB)."
echo "Models are cached in ~/.cache/mineru/ and reused on subsequent calls."
echo ""
echo "To verify installation:"
echo "  .venv/bin/mineru --version"
