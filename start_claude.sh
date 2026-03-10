#!/usr/bin/env bash
# Ensure Claude Code is on PATH inside Docker (non-login shell won't inherit ENV)
if [ -d "$HOME/.local/bin" ]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="$SCRIPT_DIR/.claude/settings.json"

PERMISSION_MODE=""
CLAUDE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --permission-mode)
      PERMISSION_MODE="$2"
      shift 2
      ;;
    *)
      CLAUDE_ARGS+=("$1")
      shift
      ;;
  esac
done

# Resolve permission mode: Docker → bypassPermissions, else use flag or default
if [ -f /.dockerenv ] || grep -q 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
  PERMISSION_MODE="bypassPermissions"
fi

PERM_ARGS=()
if [[ -n "$PERMISSION_MODE" ]]; then
  PERM_ARGS=(--permission-mode "$PERMISSION_MODE")
fi

if grep -q "minimax.io" "$SETTINGS_FILE" 2>/dev/null; then
  SECRET_FILE="/home/doanhtran03/.secrets/minimax"
  [ -f /run/secrets/minimax ] && SECRET_FILE="/run/secrets/minimax"
  ANTHROPIC_API_KEY="$(cat "$SECRET_FILE" | tr -d '[:space:]')" \
    exec claude "${PERM_ARGS[@]}" "${CLAUDE_ARGS[@]}"
else
  exec claude "${PERM_ARGS[@]}" "${CLAUDE_ARGS[@]}"
fi
