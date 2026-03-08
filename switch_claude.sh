#!/bin/bash

# Script to switch between MiniMax and default Claude authentication (project-specific)
#
# Usage:
#   ./switch_claude.sh minimax   - Switch to MiniMax
#   ./switch_claude.sh default  - Switch to default (Anthropic)
#   source ./switch_claude.sh    - Interactive mode (allows environment variable changes)
#
# NOTE: ANTHROPIC_API_KEY is set by start_claude.sh (loaded from ~/.secrets/minimax)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="$SCRIPT_DIR/.claude/settings.json"
BACKUP_FILE="$SCRIPT_DIR/.claude/settings.json.backup"

# Check if settings file exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "[error] Settings file not found at $SETTINGS_FILE"
    exit 1
fi

# Check current mode
if grep -q "minimax.io" "$SETTINGS_FILE"; then
    # Currently using MiniMax - toggle to default
    MODE="default"
else
    # Currently using default - toggle to MiniMax
    MODE="minimax"
fi

if [ "$1" = "minimax" ]; then
    MODE="minimax"
elif [ "$1" = "default" ]; then
    MODE="default"
elif [ "$1" = "status" ]; then
    if grep -q "minimax.io" "$SETTINGS_FILE"; then
        echo "[info] Current mode: MiniMax"
    else
        echo "[info] Current mode: Default (Anthropic)"
    fi
    exit 0
fi

# Backup current settings
cp "$SETTINGS_FILE" "$BACKUP_FILE"

if [ "$MODE" = "minimax" ]; then
    # Apply MiniMax configuration (API key is set by start-claude.sh at launch)
    cat > "$SETTINGS_FILE" << 'EOF'
{
    "env": {
        "ANTHROPIC_BASE_URL": "https://api.minimax.io/anthropic",
        "API_TIMEOUT_MS": "3000000",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1,
        "ANTHROPIC_MODEL": "MiniMax-M2.5",
        "ANTHROPIC_SMALL_FAST_MODEL": "MiniMax-M2.5",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "MiniMax-M2.5",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "MiniMax-M2.5",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "MiniMax-M2.5"
    }
}
EOF

    echo "[ok] Switched to MiniMax mode (project-specific)"
    echo "[info] API key will be loaded from secrets by start-claude.sh"
else
    # Apply default (Anthropic) configuration - clear auth token to use subscription
    cat > "$SETTINGS_FILE" << 'EOF'
{
    "env": {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1
    }
}
EOF

    # If sourced, clean up environment variables from the current shell
    if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
        unset ANTHROPIC_API_KEY
        unset ANTHROPIC_BASE_URL
    fi

    echo "[ok] Switched to Default (Anthropic) mode (project-specific)"
fi

echo "[info] Backup saved to: $BACKUP_FILE"
