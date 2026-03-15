#!/usr/bin/env bash
# Ensure Claude Code is on PATH inside Docker (non-login shell won't inherit ENV)
if [ -d "$HOME/.local/bin" ]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="$SCRIPT_DIR/.claude/settings.json"
BACKUP_FILE="$SCRIPT_DIR/.claude/settings.json.backup"

PERMISSION_MODE=""
PROVIDER=""
CLAUDE_ARGS=()

# Check for resume flag file (set by safe-run.sh via shared volume)
RESUME_FLAG="$SCRIPT_DIR/.claude_resume"
if [[ -f "$RESUME_FLAG" ]]; then
  CLAUDE_ARGS+=(--resume)
fi
# Always clean up the flag (trap ensures cleanup even on unexpected exit)
rm -f "$RESUME_FLAG"
trap 'rm -f "$RESUME_FLAG"' EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --permission-mode)
      PERMISSION_MODE="$2"
      shift 2
      ;;
    minimax|default)
      PROVIDER="$1"
      shift
      ;;
    *)
      CLAUDE_ARGS+=("$1")
      shift
      ;;
  esac
done

# Switch provider if requested
if [[ -n "$PROVIDER" ]]; then
  cp "$SETTINGS_FILE" "$BACKUP_FILE"
  if [[ "$PROVIDER" == "minimax" ]]; then
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
    echo "[ok] Switched to MiniMax mode"
  else
    cat > "$SETTINGS_FILE" << 'EOF'
{
    "env": {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1
    }
}
EOF
    echo "[ok] Switched to Default (Anthropic) mode"
  fi
fi

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
