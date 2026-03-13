# Citation Graph MCP Server

A Model Context Protocol (MCP) server for building academic citation graphs using the Semantic Scholar API, with an interactive web visualization interface.

## Features

- **Search papers** by keyword or Corpus ID
- **Build citation graphs** by adding papers and their connections
- **Get citations** - papers that cite a given paper (newer papers)
- **Get references** - papers that a given paper cites (older papers)
- **Auto-discovery** of connections between papers already in the graph
- **Web visualization** - Interactive D3.js force-directed graph
- **Smart toggle** - Visualization auto-starts when papers are added
- **Export graphs** in JSON, GraphML, or DOT (Graphviz) formats

## Installation

```bash
cd mcp-citation-graph
pip install -e .
```

Or install dependencies directly:

```bash
pip install mcp httpx aiohttp
```

## Quick Test (Without MCP)

Test the web visualization with sample data:

```bash
python test_web.py
```

This opens a browser with a sample citation graph.

## Usage with Claude Code

Add to your Claude Code MCP settings:

**Option 1: Direct Python**
```json
{
  "mcpServers": {
    "citation-graph": {
      "command": "python",
      "args": ["/path/to/mcp-citation-graph/server.py"]
    }
  }
}
```

**Option 2: With uv**
```json
{
  "mcpServers": {
    "citation-graph": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-citation-graph", "python", "server.py"]
    }
  }
}
```

## Available Tools

### Paper Search & Info

| Tool | Description |
|------|-------------|
| `search_papers` | Search papers by keyword (up to 50 results) |
| `get_paper` | Get detailed info about a paper |

### Graph Building

| Tool | Description |
|------|-------------|
| `add_paper_to_graph` | Add a paper as a node (auto-starts visualization) |
| `get_citations` | Get citing papers (optionally add to graph) |
| `get_references` | Get referenced papers (optionally add to graph) |
| `get_graph` | View current graph state |
| `clear_graph` | Clear all nodes and edges |
| `export_graph` | Export as JSON, GraphML, or DOT |

### Visualization Control

| Tool | Description |
|------|-------------|
| `start_visualization` | Start web server (opens browser) |
| `stop_visualization` | Stop web server |
| `visualization_status` | Check server status |

## Smart Features

### Auto-Start Visualization
When you add the first paper to the graph, the visualization server automatically starts and opens in your browser.

### Real-Time Updates
The web interface receives updates via WebSocket whenever papers are added or removed.

### Auto-Reconnect
If the connection drops, the web interface automatically attempts to reconnect with exponential backoff.

## Web Interface Features

- **Force-directed layout** - Papers automatically arrange based on citation relationships
- **Drag & drop** - Manually position papers by dragging
- **Zoom & pan** - Navigate large graphs
- **Node sizing** - Size based on citation count
- **Color coding** - Colors indicate publication year
- **Tooltips** - Hover for paper details
- **Selection** - Click to highlight connected papers
- **Export** - Download graph as JSON

## Example Workflow

```
User: Search for papers about "attention is all you need"

Claude: [Uses search_papers tool]
Found 5 papers...

User: Add the first one to the graph

Claude: [Uses add_paper_to_graph]
Added paper. Visualization auto-started at http://127.0.0.1:8765

User: Get the top 10 papers that cite it and add them

Claude: [Uses get_citations with add_to_graph=true]
Added 10 papers to graph. Graph now has 11 nodes and 10 edges.

User: Export the graph as DOT format

Claude: [Uses export_graph]
```graphviz
digraph citation_graph {
  ...
}
```
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_SCHOLAR_API_KEY` | (none) | API key for higher rate limits |
| `CITATION_GRAPH_HOST` | `127.0.0.1` | Web server host |
| `CITATION_GRAPH_PORT` | `8765` | Web server port |

### API Rate Limits

- Without API key: Shared limit of 1000 req/sec
- With API key: ~100 req/sec dedicated

Get a free API key: https://www.semanticscholar.org/product/api

## Project Structure

```
mcp-citation-graph/
├── server.py        # MCP server with web visualization
├── pyproject.toml   # Package configuration
├── test_api.py      # API test script
├── test_web.py      # Web visualization test
├── README.md        # This file
└── web/
    └── index.html   # D3.js visualization interface
```

## Export Formats

### JSON
```json
{
  "nodes": [...],
  "edges": [{"source_id": "...", "target_id": "..."}]
}
```

### GraphML
Compatible with Gephi, Cytoscape, NetworkX

### DOT (Graphviz)
```bash
# Convert to image
dot -Tpng graph.dot -o graph.png
dot -Tsvg graph.dot -o graph.svg
```

## License

MIT
