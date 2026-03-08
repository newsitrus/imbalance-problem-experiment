#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="$SCRIPT_DIR/.claude/settings.json"

if grep -q "minimax.io" "$SETTINGS_FILE" 2>/dev/null; then
  ANTHROPIC_API_KEY="$(cat /home/doanhtran03/.secrets/minimax | tr -d '[:space:]')" \
    exec claude "$@"
else
  exec claude "$@"
fi
