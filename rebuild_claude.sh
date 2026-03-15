#!/usr/bin/env bash
# Run all MCP server setup scripts inside the container.
# Usage: ./rebuild_claude.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_DIR="$SCRIPT_DIR/.claude/mcp"

if [ ! -d "$MCP_DIR" ]; then
  echo "No MCP directory found at $MCP_DIR"
  exit 1
fi

for setup in "$MCP_DIR"/*/setup.sh; do
  [ -f "$setup" ] || continue
  name="$(basename "$(dirname "$setup")")"
  echo "[mcp] Setting up $name..."
  bash "$setup" && echo "[mcp]   done: $name" || echo "[mcp]   failed: $name"
done
