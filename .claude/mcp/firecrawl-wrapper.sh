#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
export FIRECRAWL_API_KEY="$(cat /run/secrets/firecrawl)"
exec npx -y firecrawl-mcp
