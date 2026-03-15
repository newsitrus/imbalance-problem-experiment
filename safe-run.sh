#!/usr/bin/env bash

# For host to run claude in a normal docker container
# Installs Claude Code via official curl installer
#
# Usage:
#   ./safe-run.sh                  - Run with current mode
#   ./safe-run.sh minimax          - Run with MiniMax mode
#   ./safe-run.sh default          - Run with Default (Anthropic) mode
#   ./safe-run.sh --resume         - Resume the last conversation
#   ./safe-run.sh --rebuild        - Rebuild the container

WORKSPACE="/home/doanhtran03/Python/sentourism-experiment"
CONTAINER_NAME="$(basename "$(git -C "$WORKSPACE" rev-parse --show-toplevel 2>/dev/null || echo "$WORKSPACE")")-claude"
IMAGE_NAME="claude-code:latest"

# Additional volume mounts (add more as needed)
# Note: Mount Windows paths from WSL using /mnt/<drive>/...
EXTRA_MOUNTS=(
  -v "/mnt/d/Document/Research/imbalance data problem:/mnt/imbalance-data-problem"
)

# Parse arguments
PROVIDER=""
RESUME=false
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    minimax|default) PROVIDER="$arg" ;;
    --resume) RESUME=true ;;
    --rebuild) ;;
    *) PASSTHROUGH_ARGS+=("$arg") ;;
  esac
done

# Build image if it doesn't exist or --rebuild is passed
if [[ "$1" == "--rebuild" ]]; then
  shift
  docker rm -f "$CONTAINER_NAME" 2>/dev/null
  docker build -t "$IMAGE_NAME" -f "$WORKSPACE/Dockerfile.claude" "$WORKSPACE"
elif ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  docker build -t "$IMAGE_NAME" -f "$WORKSPACE/Dockerfile.claude" "$WORKSPACE"
fi

# Switch provider if requested (modifies .claude/settings.json on the host;
# the container sees it via volume mount)
if [[ -n "$PROVIDER" ]]; then
  "$WORKSPACE/switch_claude.sh" "$PROVIDER"
fi

# Signal resume via flag file (container picks it up through the shared volume)
RESUME_FLAG="$WORKSPACE/.claude_resume"
rm -f "$RESUME_FLAG"
if $RESUME; then
  touch "$RESUME_FLAG"
fi

# Reuse existing container if it exists, otherwise create one
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Starting existing container..."
  docker start -ai "$CONTAINER_NAME"
else
  echo "Creating new container..."
  docker run -it \
    --name "$CONTAINER_NAME" \
    -e TERM=xterm-256color \
    -e LC_ALL=en_US.UTF-8 \
    -e LANG=en_US.UTF-8 \
    -p 8765:8765 \
    -v "$WORKSPACE":"$WORKSPACE" \
    -v /home/doanhtran03/.secrets/minimax:/run/secrets/minimax:ro \
    -v /home/doanhtran03/.secrets/firecrawl:/run/secrets/firecrawl:ro \
    "${EXTRA_MOUNTS[@]}" \
    -w "$WORKSPACE" \
    "$IMAGE_NAME" \
    -c "./start_claude.sh --permission-mode acceptEdits ${PASSTHROUGH_ARGS[*]}"
fi
