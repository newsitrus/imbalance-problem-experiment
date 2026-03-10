#!/bin/bash
set -e
cd "$(dirname "$0")"
uv venv .venv
uv pip install --python .venv/bin/python notebooklm-mcp-cli
echo "Done. Authenticate with: .venv/bin/nlm login"
