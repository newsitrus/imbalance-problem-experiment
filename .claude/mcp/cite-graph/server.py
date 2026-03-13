#!/usr/bin/env python3
"""
Citation Graph MCP Server
A Model Context Protocol server for building citation graphs using Semantic Scholar API.
Includes an optional web visualization interface with smart toggle.
"""

import json
import asyncio
import os
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Semantic Scholar API configuration
API_BASE = "https://api.semanticscholar.org/graph/v1"
API_V1_BASE = "https://api.semanticscholar.org/v1"
PAPER_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,url,references,citations"
SEARCH_FIELDS = "paperId,title,authors,year,citationCount,venue"
EXPAND_FIELDS = "paperId,title,authors,year,citationCount,venue,isInfluential"

# Optional API key for higher rate limits
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# Web server configuration
WEB_HOST = os.environ.get("CITATION_GRAPH_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("CITATION_GRAPH_PORT", "8765"))
WEB_DIR = Path(__file__).parent / "web"

# In-memory graph storage
graph_state = {
    "nodes": {},  # paperId -> paper data
    "edges": [],  # list of {source_id, target_id}
}

# Web server state
web_server_state = {
    "running": False,
    "server": None,
    "thread": None,
    "clients": set(),  # WebSocket clients
    "auto_open": True,  # Smart auto-open browser
}


def get_headers() -> dict:
    """Get request headers with optional API key."""
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers


async def fetch_paper(paper_id: str, client: httpx.AsyncClient) -> dict | None:
    """Fetch paper details from Semantic Scholar API."""
    encoded_id = quote(paper_id, safe=":")
    url = f"{API_BASE}/paper/{encoded_id}"
    params = {"fields": PAPER_FIELDS}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params, headers=get_headers())
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"[cite-graph] Rate limited (429), retrying in {wait}s (attempt {attempt+1}/{max_retries})", file=sys.stderr)
                await asyncio.sleep(wait)
                continue
            print(f"[cite-graph] Fetch paper failed: HTTP {response.status_code} — {response.text[:200]}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[cite-graph] Fetch paper error: {e}", file=sys.stderr)
            return None
    print(f"[cite-graph] Fetch paper exhausted retries for {paper_id}", file=sys.stderr)
    return None


async def search_papers(query: str, limit: int, client: httpx.AsyncClient) -> list | str:
    """Search for papers by keyword. Returns list of papers or error string."""
    url = f"{API_BASE}/paper/search"
    params = {"query": query, "limit": limit, "fields": SEARCH_FIELDS}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params, headers=get_headers())
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"[cite-graph] Rate limited (429), retrying in {wait}s (attempt {attempt+1}/{max_retries})", file=sys.stderr)
                await asyncio.sleep(wait)
                continue
            print(f"[cite-graph] Search failed: HTTP {response.status_code} — {response.text[:200]}", file=sys.stderr)
            return f"API error: HTTP {response.status_code}"
        except Exception as e:
            print(f"[cite-graph] Search error: {e}", file=sys.stderr)
            return f"Request error: {e}"
    return "Rate limited by Semantic Scholar API. Try again later or set SEMANTIC_SCHOLAR_API_KEY env var for higher limits."


async def fetch_sorted_neighbors(
    paper_id: str, direction: str, sort: str, limit: int, client: httpx.AsyncClient
) -> list[dict] | str:
    """Fetch citations or references, sorted by the given criteria.
    direction: 'citations' or 'references'
    sort: 'citation-count', 'year', 'influential', 'relevance'
    Returns list of paper dicts (with paperId, title, authors, year, citationCount, isInfluential)
    or an error string.
    """
    encoded_id = quote(paper_id, safe=":")
    # Fetch more than needed so we can deduplicate against existing graph nodes
    fetch_limit = min(limit * 5, 100)
    url = f"{API_BASE}/paper/{encoded_id}/{direction}"
    params = {"fields": EXPAND_FIELDS, "limit": fetch_limit}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.get(url, params=params, headers=get_headers())
            if response.status_code == 200:
                break
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"[cite-graph] Rate limited (429), retrying in {wait}s", file=sys.stderr)
                await asyncio.sleep(wait)
                continue
            return f"API error: HTTP {response.status_code}"
        except Exception as e:
            return f"Request error: {e}"
    else:
        return "Rate limited by Semantic Scholar API. Try again later."

    data = response.json()
    items = data.get("data", [])

    # Extract the nested paper object
    key = "citingPaper" if direction == "citations" else "citedPaper"
    papers = []
    for item in items:
        p = item.get(key, {})
        if not p or not p.get("paperId"):
            continue
        p["isInfluential"] = item.get("isInfluential", False)
        papers.append(p)

    # Filter out papers already in the graph
    existing_ids = set(graph_state["nodes"].keys())
    papers = [p for p in papers if p["paperId"] not in existing_ids]

    # Sort
    if sort == "citation-count":
        papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    elif sort == "year":
        papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
    elif sort == "influential":
        papers.sort(key=lambda p: (p.get("isInfluential", False), p.get("citationCount") or 0), reverse=True)
    # 'relevance' = default API order, no re-sort needed

    return papers[:limit]


def add_node_to_graph(paper: dict, parent_id: str | None = None, is_citation: bool = True) -> dict:
    """Add a paper node to the graph."""
    paper_id = paper.get("paperId")
    if not paper_id:
        return {"success": False, "error": "No paperId"}

    # Add node if not exists
    if paper_id not in graph_state["nodes"]:
        graph_state["nodes"][paper_id] = {
            "id": paper_id,
            "title": paper.get("title", "Unknown"),
            "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
            "year": paper.get("year"),
            "citationCount": paper.get("citationCount", 0),
            "venue": paper.get("venue", ""),
            "url": paper.get("url", ""),
            "abstract": paper.get("abstract", ""),
        }

    # Add edge if parent exists
    if parent_id and parent_id in graph_state["nodes"]:
        if is_citation:
            edge = {"source_id": paper_id, "target_id": parent_id}
        else:
            edge = {"source_id": parent_id, "target_id": paper_id}

        if edge not in graph_state["edges"]:
            graph_state["edges"].append(edge)

    # Auto-discover connections to existing nodes
    for citation in paper.get("citations", []) or []:
        cit_id = citation.get("paperId")
        if cit_id and cit_id in graph_state["nodes"]:
            edge = {"source_id": cit_id, "target_id": paper_id}
            if edge not in graph_state["edges"]:
                graph_state["edges"].append(edge)

    for reference in paper.get("references", []) or []:
        ref_id = reference.get("paperId")
        if ref_id and ref_id in graph_state["nodes"]:
            edge = {"source_id": paper_id, "target_id": ref_id}
            if edge not in graph_state["edges"]:
                graph_state["edges"].append(edge)

    # Notify web clients of update
    broadcast_graph_update()

    return {"success": True, "paperId": paper_id}


def get_graph_data() -> dict:
    """Get graph data for web interface."""
    return {
        "nodes": list(graph_state["nodes"].values()),
        "edges": graph_state["edges"]
    }


def broadcast_graph_update():
    """Broadcast graph update to all connected WebSocket clients."""
    if not web_server_state["running"]:
        return

    try:
        # This will be handled by the web server's event loop
        pass
    except Exception as e:
        pass


# ============== Web Server (Optional) ==============

def start_web_server():
    """Start the web visualization server."""
    try:
        from aiohttp import web
        import aiohttp
    except ImportError:
        return {"success": False, "error": "aiohttp not installed. Run: pip install aiohttp"}

    if web_server_state["running"]:
        return {"success": True, "message": "Server already running", "url": f"http://{WEB_HOST}:{WEB_PORT}"}

    async def handle_index(request):
        """Serve the main HTML page."""
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="Web interface not found", status=404)

    async def handle_api_graph(request):
        """API endpoint to get current graph."""
        return web.json_response(get_graph_data())

    async def handle_api_search(request):
        """API endpoint to search papers from the web UI."""
        query = request.query.get("query", "")
        limit = min(int(request.query.get("limit", "10")), 50)
        if not query:
            return web.json_response({"error": "Missing query parameter"}, status=400)
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await search_papers(query, limit, client)
        if isinstance(result, str):
            return web.json_response({"error": result})
        return web.json_response({"data": result})

    async def handle_api_add_paper(request):
        """API endpoint to add a paper to the graph from the web UI."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)
        paper_id = body.get("paper_id", "")
        if not paper_id:
            return web.json_response({"success": False, "error": "Missing paper_id"}, status=400)
        async with httpx.AsyncClient(timeout=30.0) as client:
            paper = await fetch_paper(paper_id, client)
        if not paper:
            return web.json_response({"success": False, "error": f"Paper not found: {paper_id}"})
        result = add_node_to_graph(paper)
        if result["success"]:
            # Notify WebSocket clients of the update
            graph_update = {"type": "graph_update", "graph": get_graph_data()}
            for ws_client in list(web_server_state["clients"]):
                try:
                    await ws_client.send_json(graph_update)
                except Exception:
                    web_server_state["clients"].discard(ws_client)
        return web.json_response(result)

    async def handle_api_expand(request):
        """API endpoint to expand a node with sorted citations or references."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid request body"}, status=400)
        paper_id = body.get("paper_id", "")
        direction = body.get("direction", "citations")  # 'citations' or 'references'
        sort = body.get("sort", "citation-count")
        limit = min(int(body.get("limit", 5)), 20)
        if not paper_id:
            return web.json_response({"success": False, "error": "Missing paper_id"}, status=400)
        if direction not in ("citations", "references"):
            return web.json_response({"success": False, "error": "direction must be 'citations' or 'references'"}, status=400)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                result = await fetch_sorted_neighbors(paper_id, direction, sort, limit, client)
                if isinstance(result, str):
                    return web.json_response({"success": False, "error": result})

                if not result:
                    return web.json_response({
                        "success": True,
                        "added": [],
                        "direction": direction,
                        "sort": sort,
                        "graph": get_graph_data(),
                        "message": f"No new {direction} found (all already in graph or none available)",
                    })

                added = []
                failed = []
                for p in result:
                    full_paper = await fetch_paper(p["paperId"], client)
                    if full_paper:
                        is_citation = direction == "citations"
                        add_node_to_graph(full_paper, paper_id, is_citation=is_citation)
                        added.append({
                            "paperId": p["paperId"],
                            "title": p.get("title", "Unknown"),
                            "year": p.get("year"),
                            "citationCount": p.get("citationCount", 0),
                        })
                    else:
                        failed.append(p.get("title", p["paperId"]))

            # Broadcast update to WebSocket clients
            graph_update = {"type": "graph_update", "graph": get_graph_data()}
            for ws_client in list(web_server_state["clients"]):
                try:
                    await ws_client.send_json(graph_update)
                except Exception:
                    web_server_state["clients"].discard(ws_client)

            resp = {
                "success": True,
                "added": added,
                "direction": direction,
                "sort": sort,
                "graph": get_graph_data(),
            }
            if failed:
                resp["warning"] = f"Could not fetch details for {len(failed)} paper(s): {', '.join(failed[:3])}"
            return web.json_response(resp)

        except asyncio.TimeoutError:
            return web.json_response({"success": False, "error": "Request timed out. Semantic Scholar API may be slow — try again."})
        except Exception as e:
            print(f"[cite-graph] Expand error: {e}", file=sys.stderr)
            return web.json_response({"success": False, "error": f"Server error: {str(e)}"})

    async def handle_websocket(request):
        """WebSocket handler for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        web_server_state["clients"].add(ws)
        print(f"WebSocket client connected. Total: {len(web_server_state['clients'])}")

        try:
            # Send current graph state
            await ws.send_json({"type": "graph_update", "graph": get_graph_data()})

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            web_server_state["clients"].discard(ws)
            print(f"WebSocket client disconnected. Total: {len(web_server_state['clients'])}")

        return ws

    async def broadcast_to_clients(data: dict):
        """Broadcast data to all connected clients."""
        for ws in list(web_server_state["clients"]):
            try:
                await ws.send_json(data)
            except Exception:
                web_server_state["clients"].discard(ws)

    def run_server():
        """Run the web server in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = web.Application()
        app.router.add_get("/", handle_index)
        app.router.add_get("/api/graph", handle_api_graph)
        app.router.add_get("/api/search", handle_api_search)
        app.router.add_post("/api/add_paper", handle_api_add_paper)
        app.router.add_post("/api/expand", handle_api_expand)
        app.router.add_get("/ws", handle_websocket)

        # Store broadcast function
        app["broadcast"] = broadcast_to_clients

        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())

        site = web.TCPSite(runner, WEB_HOST, WEB_PORT)
        loop.run_until_complete(site.start())

        web_server_state["running"] = True
        web_server_state["server"] = runner
        web_server_state["loop"] = loop

        print(f"Web server started at http://{WEB_HOST}:{WEB_PORT}")

        loop.run_forever()

    # Start server in background thread
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    web_server_state["thread"] = thread

    # Wait a moment for server to start
    import time
    time.sleep(0.5)

    url = f"http://{WEB_HOST}:{WEB_PORT}"

    # Smart auto-open browser
    if web_server_state["auto_open"]:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return {"success": True, "url": url}


def stop_web_server():
    """Stop the web visualization server."""
    if not web_server_state["running"]:
        return {"success": True, "message": "Server not running"}

    web_server_state["running"] = False

    if web_server_state.get("loop"):
        web_server_state["loop"].call_soon_threadsafe(web_server_state["loop"].stop)

    return {"success": True, "message": "Server stopped"}


def get_web_status():
    """Get web server status."""
    return {
        "running": web_server_state["running"],
        "url": f"http://{WEB_HOST}:{WEB_PORT}" if web_server_state["running"] else None,
        "clients": len(web_server_state["clients"]),
        "auto_open": web_server_state["auto_open"]
    }


# ============== MCP Server ==============

server = Server("citation-graph")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_papers",
            description="Search for academic papers by keyword using Semantic Scholar API. Returns up to 50 papers matching the query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keywords, paper title, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (1-50, default 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_paper",
            description="Get detailed information about a specific paper by its Semantic Scholar Paper ID or Corpus ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Semantic Scholar Paper ID or Corpus ID (e.g., 'CorpusID:12345')"
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="add_paper_to_graph",
            description="Add a paper to the citation graph. Fetches paper details and adds it as a node. Automatically discovers connections to existing nodes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Semantic Scholar Paper ID to add"
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="get_citations",
            description="Get papers that CITE the specified paper (newer papers that reference this one). Optionally add them to the graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper ID to get citations for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of citations to return (1-100, default 10)",
                        "default": 10
                    },
                    "add_to_graph": {
                        "type": "boolean",
                        "description": "Whether to add the citations to the graph (default false)",
                        "default": False
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="get_references",
            description="Get papers that the specified paper REFERENCES (older papers cited by this one). Optionally add them to the graph.",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "Paper ID to get references for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of references to return (1-100, default 10)",
                        "default": 10
                    },
                    "add_to_graph": {
                        "type": "boolean",
                        "description": "Whether to add the references to the graph (default false)",
                        "default": False
                    }
                },
                "required": ["paper_id"]
            }
        ),
        Tool(
            name="get_graph",
            description="Get the current state of the citation graph, including all nodes (papers) and edges (citation relationships).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="clear_graph",
            description="Clear all nodes and edges from the citation graph.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="export_graph",
            description="Export the citation graph in various formats (JSON, GraphML, or DOT for Graphviz).",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["json", "graphml", "dot"],
                        "description": "Export format (default: json)",
                        "default": "json"
                    }
                }
            }
        ),
        Tool(
            name="start_visualization",
            description="Start the web visualization server to view the citation graph in a browser. Opens automatically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_open": {
                        "type": "boolean",
                        "description": "Automatically open browser (default true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="stop_visualization",
            description="Stop the web visualization server.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="visualization_status",
            description="Get the current status of the web visualization server.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    async with httpx.AsyncClient(timeout=30.0) as client:

        if name == "search_papers":
            query = arguments.get("query", "")
            limit = min(max(arguments.get("limit", 10), 1), 50)

            result_or_error = await search_papers(query, limit, client)

            if isinstance(result_or_error, str):
                return [TextContent(type="text", text=result_or_error)]

            papers = result_or_error
            if not papers:
                return [TextContent(type="text", text=f"No papers found for query: {query}")]

            result = f"Found {len(papers)} papers:\n\n"
            for i, paper in enumerate(papers, 1):
                authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])][:3])
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."
                result += f"{i}. **{paper.get('title', 'Unknown')}**\n"
                result += f"   - Paper ID: `{paper.get('paperId')}`\n"
                result += f"   - Authors: {authors}\n"
                result += f"   - Year: {paper.get('year', 'N/A')}\n"
                result += f"   - Citations: {paper.get('citationCount', 0)}\n\n"

            return [TextContent(type="text", text=result)]

        elif name == "get_paper":
            paper_id = arguments.get("paper_id", "")
            paper = await fetch_paper(paper_id, client)

            if not paper:
                return [TextContent(type="text", text=f"Paper not found: {paper_id}")]

            authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
            citations_list = paper.get("citations", []) or []
            references_list = paper.get("references", []) or []

            result = f"# {paper.get('title', 'Unknown')}\n\n"
            result += f"**Paper ID:** `{paper.get('paperId')}`\n"
            result += f"**Authors:** {authors}\n"
            result += f"**Year:** {paper.get('year', 'N/A')}\n"
            result += f"**Venue:** {paper.get('venue', 'N/A')}\n"
            result += f"**Citation Count:** {paper.get('citationCount', 0)}\n"
            result += f"**URL:** {paper.get('url', 'N/A')}\n\n"
            result += f"**Abstract:**\n{paper.get('abstract', 'No abstract available')}\n\n"
            result += f"**Citations (papers citing this):** {len(citations_list)}\n"
            result += f"**References (papers this cites):** {len(references_list)}\n"

            return [TextContent(type="text", text=result)]

        elif name == "add_paper_to_graph":
            paper_id = arguments.get("paper_id", "")
            paper = await fetch_paper(paper_id, client)

            if not paper:
                return [TextContent(type="text", text=f"Paper not found: {paper_id}")]

            result = add_node_to_graph(paper)

            if result["success"]:
                node_count = len(graph_state["nodes"])
                edge_count = len(graph_state["edges"])

                # Smart auto-start visualization if not running and we have nodes
                status_msg = ""
                if node_count >= 1 and not web_server_state["running"]:
                    start_result = start_web_server()
                    if start_result["success"]:
                        status_msg = f"\n\n🌐 Visualization auto-started: {start_result['url']}"

                return [TextContent(type="text", text=f"Added paper '{paper.get('title')}' to graph.\n\nGraph now has {node_count} nodes and {edge_count} edges.{status_msg}")]
            else:
                return [TextContent(type="text", text=f"Failed to add paper: {result.get('error')}")]

        elif name == "get_citations":
            paper_id = arguments.get("paper_id", "")
            limit = min(max(arguments.get("limit", 10), 1), 100)
            add_to_graph = arguments.get("add_to_graph", False)

            encoded_id = quote(paper_id, safe=":")
            url = f"{API_BASE}/paper/{encoded_id}/citations"
            params = {"fields": SEARCH_FIELDS, "limit": limit}
            try:
                response = await client.get(url, params=params, headers=get_headers())
                if response.status_code != 200:
                    return [TextContent(type="text", text=f"Failed to get citations for paper: {paper_id} (HTTP {response.status_code})")]

                data = response.json()
                citations = data.get("data", [])
            except Exception as e:
                return [TextContent(type="text", text=f"Error fetching citations: {str(e)}")]

            if not citations:
                return [TextContent(type="text", text=f"No citations found for paper: {paper_id}")]

            result = f"Found {len(citations)} citations (papers that cite this paper):\n\n"
            added_count = 0

            for i, item in enumerate(citations, 1):
                citing_paper = item.get("citingPaper", {})
                authors = ", ".join([a.get("name", "") for a in citing_paper.get("authors", [])][:2])
                if len(citing_paper.get("authors", [])) > 2:
                    authors += " et al."

                result += f"{i}. **{citing_paper.get('title', 'Unknown')}**\n"
                result += f"   - Paper ID: `{citing_paper.get('paperId')}`\n"
                result += f"   - Authors: {authors}\n"
                result += f"   - Year: {citing_paper.get('year', 'N/A')}\n"
                result += f"   - Citations: {citing_paper.get('citationCount', 0)}\n\n"

                if add_to_graph and citing_paper.get("paperId"):
                    full_paper = await fetch_paper(citing_paper["paperId"], client)
                    if full_paper:
                        add_node_to_graph(full_paper, paper_id, is_citation=True)
                        added_count += 1

            if add_to_graph:
                result += f"\n---\nAdded {added_count} papers to graph. Graph now has {len(graph_state['nodes'])} nodes and {len(graph_state['edges'])} edges."

            return [TextContent(type="text", text=result)]

        elif name == "get_references":
            paper_id = arguments.get("paper_id", "")
            limit = min(max(arguments.get("limit", 10), 1), 100)
            add_to_graph = arguments.get("add_to_graph", False)

            encoded_id = quote(paper_id, safe=":")
            url = f"{API_BASE}/paper/{encoded_id}/references"
            params = {"fields": SEARCH_FIELDS, "limit": limit}
            try:
                response = await client.get(url, params=params, headers=get_headers())
                if response.status_code != 200:
                    return [TextContent(type="text", text=f"Failed to get references for paper: {paper_id} (HTTP {response.status_code})")]

                data = response.json()
                references = data.get("data", [])
            except Exception as e:
                return [TextContent(type="text", text=f"Error fetching references: {str(e)}")]

            if not references:
                return [TextContent(type="text", text=f"No references found for paper: {paper_id}")]

            result = f"Found {len(references)} references (papers cited by this paper):\n\n"
            added_count = 0

            for i, item in enumerate(references, 1):
                cited_paper = item.get("citedPaper", {})
                authors = ", ".join([a.get("name", "") for a in cited_paper.get("authors", [])][:2])
                if len(cited_paper.get("authors", [])) > 2:
                    authors += " et al."

                result += f"{i}. **{cited_paper.get('title', 'Unknown')}**\n"
                result += f"   - Paper ID: `{cited_paper.get('paperId')}`\n"
                result += f"   - Authors: {authors}\n"
                result += f"   - Year: {cited_paper.get('year', 'N/A')}\n"
                result += f"   - Citations: {cited_paper.get('citationCount', 0)}\n\n"

                if add_to_graph and cited_paper.get("paperId"):
                    full_paper = await fetch_paper(cited_paper["paperId"], client)
                    if full_paper:
                        add_node_to_graph(full_paper, paper_id, is_citation=False)
                        added_count += 1

            if add_to_graph:
                result += f"\n---\nAdded {added_count} papers to graph. Graph now has {len(graph_state['nodes'])} nodes and {len(graph_state['edges'])} edges."

            return [TextContent(type="text", text=result)]

        elif name == "get_graph":
            nodes = graph_state["nodes"]
            edges = graph_state["edges"]

            if not nodes:
                return [TextContent(type="text", text="Graph is empty. Use 'add_paper_to_graph' or 'get_citations'/'get_references' with add_to_graph=true to add papers.")]

            result = f"# Citation Graph\n\n"
            result += f"**Nodes:** {len(nodes)} papers\n"
            result += f"**Edges:** {len(edges)} citation relationships\n\n"

            result += "## Papers in Graph\n\n"
            for paper_id, paper in nodes.items():
                result += f"- **{paper['title']}** ({paper['year'] or 'N/A'})\n"
                result += f"  - ID: `{paper_id}`\n"
                result += f"  - Authors: {paper['authors'][:50]}{'...' if len(paper['authors']) > 50 else ''}\n"
                result += f"  - Citations: {paper['citationCount']}\n\n"

            if edges:
                result += "## Citation Relationships\n\n"
                for edge in edges[:20]:
                    source = nodes.get(edge["source_id"], {}).get("title", edge["source_id"])[:40]
                    target = nodes.get(edge["target_id"], {}).get("title", edge["target_id"])[:40]
                    result += f"- {source}... → {target}...\n"

                if len(edges) > 20:
                    result += f"\n... and {len(edges) - 20} more edges\n"

            return [TextContent(type="text", text=result)]

        elif name == "clear_graph":
            graph_state["nodes"] = {}
            graph_state["edges"] = []
            return [TextContent(type="text", text="Graph cleared successfully.")]

        elif name == "export_graph":
            format_type = arguments.get("format", "json")
            nodes = graph_state["nodes"]
            edges = graph_state["edges"]

            if format_type == "json":
                export_data = {
                    "nodes": list(nodes.values()),
                    "edges": edges
                }
                result = json.dumps(export_data, indent=2)
                return [TextContent(type="text", text=f"```json\n{result}\n```")]

            elif format_type == "graphml":
                result = '<?xml version="1.0" encoding="UTF-8"?>\n'
                result += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
                result += '  <key id="title" for="node" attr.name="title" attr.type="string"/>\n'
                result += '  <key id="year" for="node" attr.name="year" attr.type="int"/>\n'
                result += '  <key id="citations" for="node" attr.name="citations" attr.type="int"/>\n'
                result += '  <graph id="citation_graph" edgedefault="directed">\n'

                for paper_id, paper in nodes.items():
                    title_escaped = paper["title"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    result += f'    <node id="{paper_id}">\n'
                    result += f'      <data key="title">{title_escaped}</data>\n'
                    result += f'      <data key="year">{paper["year"] or 0}</data>\n'
                    result += f'      <data key="citations">{paper["citationCount"]}</data>\n'
                    result += '    </node>\n'

                for i, edge in enumerate(edges):
                    result += f'    <edge id="e{i}" source="{edge["source_id"]}" target="{edge["target_id"]}"/>\n'

                result += '  </graph>\n</graphml>'
                return [TextContent(type="text", text=f"```xml\n{result}\n```")]

            elif format_type == "dot":
                result = "digraph citation_graph {\n"
                result += "  rankdir=LR;\n"
                result += "  node [shape=box];\n\n"

                for paper_id, paper in nodes.items():
                    label = paper["title"][:30].replace('"', '\\"')
                    year = paper["year"] or "N/A"
                    result += f'  "{paper_id}" [label="{label}...\\n({year})"];\n'

                result += "\n"
                for edge in edges:
                    result += f'  "{edge["source_id"]}" -> "{edge["target_id"]}";\n'

                result += "}"
                return [TextContent(type="text", text=f"```dot\n{result}\n```")]

            else:
                return [TextContent(type="text", text=f"Unknown format: {format_type}")]

        elif name == "start_visualization":
            auto_open = arguments.get("auto_open", True)
            web_server_state["auto_open"] = auto_open

            result = start_web_server()

            if result["success"]:
                return [TextContent(type="text", text=f"🌐 Visualization server started!\n\nURL: {result['url']}\n\nThe graph visualization is now available in your browser.")]
            else:
                return [TextContent(type="text", text=f"Failed to start visualization: {result.get('error', 'Unknown error')}\n\nTry: pip install aiohttp")]

        elif name == "stop_visualization":
            result = stop_web_server()
            return [TextContent(type="text", text=f"Visualization server stopped.")]

        elif name == "visualization_status":
            status = get_web_status()
            if status["running"]:
                return [TextContent(type="text", text=f"🟢 Visualization server is running\n\nURL: {status['url']}\nConnected clients: {status['clients']}\nAuto-open browser: {status['auto_open']}")]
            else:
                return [TextContent(type="text", text=f"🔴 Visualization server is not running\n\nUse 'start_visualization' to start it.")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
