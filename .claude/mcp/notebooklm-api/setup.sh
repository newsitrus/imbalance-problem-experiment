#!/bin/bash
set -e
cd "$(dirname "$0")"

# Pin Python version so the venv works regardless of which OS user runs it.
# uv resolves --python to a user-local pool (/home/<user>/.local/share/uv/python/),
# so an unpinned "uv venv" may resolve to different patch versions per user,
# and the resulting symlink breaks when another user tries to use the venv.
PYTHON_VERSION="3.12"
uv python install "$PYTHON_VERSION"
uv venv .venv --python "$PYTHON_VERSION"
uv pip install --python .venv/bin/python notebooklm-mcp-cli
echo "Done. Authenticate with: .venv/bin/nlm login"
