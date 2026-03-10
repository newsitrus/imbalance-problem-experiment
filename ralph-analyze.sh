#!/usr/bin/env bash
set -euo pipefail

# Ralph-style loop: audit papers, then analyze until all complete.
# Reuses the container created by safe-run.sh.
# Usage: ./ralph-analyze.sh [max_iterations]

MAX_ITER="${1:-10}"
WORKSPACE="/home/doanhtran03/Python/sentourism-experiment"
CONTAINER_NAME="$(basename "$(git -C "$WORKSPACE" rev-parse --show-toplevel 2>/dev/null || echo "$WORKSPACE")")-claude"

# Check that the container exists (created by safe-run.sh)
if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Error: Container '$CONTAINER_NAME' not found."
  echo "Run ./safe-run.sh first to create the environment, then re-run this script."
  exit 1
fi

# Ensure the container is running
if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]; then
  echo "Container is stopped. Starting it..."
  docker start "$CONTAINER_NAME"
fi

PROMPT='You are running inside an automated loop. Your job:

1. Run /audit-papers (no arguments — audit ALL paper folders).
2. Read the summary. If ALL papers have every image marked "yes" in the "In content.md" column, output exactly </complete> and stop.
3. Otherwise, pick the first paper that has images with "In content.md: no" and run /paper-analyzer on that paper folder to analyze its unprocessed images.
4. After /paper-analyzer finishes, run /audit-papers again on that paper to update its _status.md.

Rules:
- Process only ONE paper per iteration. The loop will call you again for the next one.
- If /paper-analyzer or /audit-papers is not available as a skill, invoke the instructions from the SKILL.md files directly.
- When all papers are fully analyzed (all images referenced in content.md), output </complete> as the very last thing you print.'

for ((i=1; i<=MAX_ITER; i++)); do
  printf "\n\n============ Iteration %s of %s ============\n\n" "$i" "$MAX_ITER"

  result=$(docker exec \
    -e TERM=xterm-256color \
    -w "$WORKSPACE" \
    "$CONTAINER_NAME" \
    bash -c "export PATH=\"/home/claude/.local/bin:\$PATH\" && claude --permission-mode bypassPermissions -p '${PROMPT//\'/\'\\\'\'}'" \
    2>&1) || true

  printf "%s\n" "$result"

  if [[ "$result" == *"</complete>"* ]]; then
    printf "\n============ All papers analyzed. Exiting after %s iterations. ============\n" "$i"
    exit 0
  fi
done

printf "\n============ Reached max iterations (%s) without completing. ============\n" "$MAX_ITER"
exit 1
