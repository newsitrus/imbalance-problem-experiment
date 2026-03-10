#!/usr/bin/env bash
SECRET_FILE="/home/doanhtran03/.secrets/minimax"
[ -f /run/secrets/minimax ] && SECRET_FILE="/run/secrets/minimax"
export MINIMAX_API_KEY="$(cat "$SECRET_FILE" | tr -d '[:space:]')"
export MINIMAX_API_HOST="https://api.minimax.io"
exec uvx minimax-coding-plan-mcp -y
