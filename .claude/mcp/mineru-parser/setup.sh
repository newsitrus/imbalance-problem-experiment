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

# Pin Python version so the venv works regardless of which OS user runs it.
# uv resolves --python to a user-local pool (/home/<user>/.local/share/uv/python/),
# so "python3" may resolve to different patch versions per user, and the
# resulting symlink breaks when another user tries to use the venv.
PYTHON_VERSION="3.12"

echo "Ensuring Python $PYTHON_VERSION is available..."
uv python install "$PYTHON_VERSION"

# Create fresh virtual environment
echo "Creating virtual environment..."
uv venv .venv --python "$PYTHON_VERSION"

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
