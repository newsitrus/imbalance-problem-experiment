#!/usr/bin/env python3
"""
Citation Graph MCP Server
A Model Context Protocol server for building citation graphs using Semantic Scholar API.
Includes an optional web visualization interface with smart toggle.
"""

import json
import asyncio
import concurrent.futures
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
INTERNAL_BASE = "https://www.semanticscholar.org/api/1"
PAPER_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,url,references,citations"
SEARCH_FIELDS = "paperId,title,authors,year,citationCount,venue"

# Sort value mapping: user-facing name -> internal API sort value
SORT_MAP = {
    "relevance": "relevance",
    "citation-count": "total-citations",
    "year": "pub-date",
    "influential": "is-influential",
}

# Optional API key for higher rate limits
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# Web server configuration
WEB_HOST = os.environ.get("CITATION_GRAPH_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("CITATION_GRAPH_PORT", "8765"))
WEB_DIR = Path(__file__).parent / "web"

# Session cookies for S2 internal API (used by citations/references via httpx)
_s2_cookies: dict = {}

# Dedicated single-thread executor for Playwright (greenlet-safe)
_pw_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
_pw_instance = None
_pw_browser = None
_pw_context = None


# In-memory graph storage
graph_state = {
    "nodes": {},  # paperId -> paper data
    "edges": [],  # list of {source_id, target_id}
}

# Track expand pagination: (paper_id, direction, sort) -> next page number
_expand_pages: dict[tuple[str, str, str], int] = {}

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
    """Fetch paper details via Playwright browser (internal API)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pw_executor, _pw_fetch_paper, paper_id)


def _parse_search_results(results: list) -> list[dict]:
    """Normalize internal search API results into simple paper dicts."""
    papers = []
    for item in results:
        pid = item.get("id")
        if not pid:
            continue
        title = item.get("title", {})
        title_text = title.get("text", "Unknown") if isinstance(title, dict) else str(title)
        authors_raw = item.get("authors", [])
        author_list = []
        for a in authors_raw:
            if isinstance(a, list) and len(a) > 0:
                author_list.append({"name": a[0].get("name", "")})
            elif isinstance(a, dict):
                author_list.append({"name": a.get("name", "")})
        venue = item.get("venue")
        venue_text = venue.get("text", "") if isinstance(venue, dict) else str(venue or "")
        year = item.get("year")
        year_val = year.get("text", None) if isinstance(year, dict) else year
        try:
            year_val = int(year_val) if year_val else None
        except (ValueError, TypeError):
            year_val = None
        # Citation count: try numCitedBy first, then scorecardStats, then citationStats
        cite_count = item.get("numCitedBy")
        if not cite_count:
            scorecard = item.get("scorecardStats", [])
            if scorecard and isinstance(scorecard, list):
                cite_count = scorecard[0].get("citationCount", 0)
        if not cite_count:
            cite_count = (item.get("citationStats") or {}).get("numCitations", 0)
        papers.append({
            "paperId": pid,
            "title": title_text,
            "authors": author_list,
            "year": year_val,
            "citationCount": cite_count or 0,
            "venue": venue_text,
        })
    return papers


async def search_papers(query: str, limit: int, client: httpx.AsyncClient) -> list | str:
    """Search papers via Playwright browser (WAF requires real browser TLS for search)."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pw_executor, _pw_search, query, limit)
    if isinstance(result, str):
        return result
    return _parse_search_results(result)


_BROWSER_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://www.semanticscholar.org",
    "Referer": "https://www.semanticscholar.org/",
}


def _get_pw_context():
    """Get or create a persistent Playwright browser context. Must run on _pw_executor thread."""
    global _pw_instance, _pw_browser, _pw_context
    if _pw_context is not None:
        return _pw_context
    import time
    from playwright.sync_api import sync_playwright
    _pw_instance = sync_playwright().start()
    _pw_browser = _pw_instance.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled"],
    )
    _pw_context = _pw_browser.new_context(
        user_agent=_BROWSER_HEADERS["User-Agent"],
    )
    # Warm up: navigate to S2 to establish WAF cookies
    page = _pw_context.new_page()
    page.add_init_script('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')
    page.goto(
        "https://www.semanticscholar.org/search?q=test&sort=relevance",
        wait_until="networkidle", timeout=30000,
    )
    time.sleep(3)
    # Extract cookies for httpx (used by citations/references)
    for c in _pw_context.cookies():
        _s2_cookies[c["name"]] = c["value"]
    page.close()
    has_waf = "aws-waf-token" in _s2_cookies
    print(f"[cite-graph] Playwright browser ready, {len(_s2_cookies)} cookie(s), aws-waf-token={has_waf}", file=sys.stderr)
    return _pw_context


def _pw_search(query: str, limit: int) -> list[dict] | str:
    """Search S2 via Playwright: navigate to search URL and intercept API response."""
    import time
    from urllib.parse import quote as urlquote
    ctx = _get_pw_context()
    page = ctx.new_page()
    page.add_init_script('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')

    search_data = []

    def on_response(response):
        if "/api/1/search" in response.url and response.request.method == "POST":
            try:
                body = response.json()
                if "results" in body:
                    search_data.append(body)
            except Exception:
                pass

    page.on("response", on_response)
    url = f"https://www.semanticscholar.org/search?q={urlquote(query)}&sort=relevance"
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(2)
    except Exception as e:
        page.close()
        return f"Browser navigation error: {e}"
    page.close()

    if not search_data:
        return "No search results intercepted. Try again."
    return search_data[0].get("results", [])[:limit]


_JS_FETCH_PAPER = """
async (pid) => {
    const resp = await fetch("https://www.semanticscholar.org/api/1/paper/" + pid, {headers: {"Accept": "application/json"}});
    if (!resp.ok) return {error: "HTTP " + resp.status};
    const data = await resp.json();
    const p = data.paper;
    if (!p) return {error: "No paper in response"};

    const title = p.title && p.title.text ? p.title.text : (typeof p.title === "string" ? p.title : "Unknown");
    const abs = p.paperAbstract && p.paperAbstract.text ? p.paperAbstract.text : "";
    const year = p.year && p.year.text ? parseInt(p.year.text) : (typeof p.year === "number" ? p.year : null);
    const venue = p.venue && p.venue.text ? p.venue.text : (typeof p.venue === "string" ? p.venue : "");
    const url = p.primaryPaperLink ? p.primaryPaperLink.url : "";

    let citationCount = p.numCitedBy || 0;
    if (!citationCount && p.scorecardStats && p.scorecardStats.length > 0) {
        citationCount = p.scorecardStats[0].citationCount || 0;
    }

    const authors = (p.authors || []).map(a => {
        if (Array.isArray(a) && a.length > 0) return {name: a[0].name || ""};
        if (a && a.name) return {name: a.name};
        return {name: ""};
    });

    // Collect reference/citation IDs for auto-discovery
    const refs = (data.citedPapers && data.citedPapers.citations || []).map(c => ({paperId: c.id}));
    const cites = (data.citingPapers && data.citingPapers.citations || []).map(c => ({paperId: c.id}));

    return {
        paperId: p.id,
        title: title,
        abstract: abs,
        authors: authors,
        year: year,
        citationCount: citationCount,
        venue: venue,
        url: url || ("https://www.semanticscholar.org/paper/" + p.id),
        references: refs,
        citations: cites,
    };
}
"""


_pw_fetch_page = None  # Reusable page on S2 domain for API fetches


def _get_fetch_page():
    """Get a reusable Playwright page on the S2 domain for API fetches."""
    global _pw_fetch_page
    if _pw_fetch_page is not None and not _pw_fetch_page.is_closed():
        return _pw_fetch_page
    ctx = _get_pw_context()
    _pw_fetch_page = ctx.new_page()
    _pw_fetch_page.add_init_script('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')
    _pw_fetch_page.goto("https://www.semanticscholar.org/", wait_until="commit", timeout=15000)
    return _pw_fetch_page


def _pw_fetch_paper(paper_id: str) -> dict | None:
    """Fetch paper details via Playwright in-browser fetch to internal API."""
    page = _get_fetch_page()
    try:
        result = page.evaluate(_JS_FETCH_PAPER, paper_id)
    except Exception as e:
        print(f"[cite-graph] Playwright fetch_paper error: {e}", file=sys.stderr)
        return None
    if isinstance(result, dict) and "error" in result:
        print(f"[cite-graph] Internal paper API: {result['error']}", file=sys.stderr)
        return None
    return result


def _parse_internal_results(results: list) -> list[dict]:
    """Normalize the internal API response into simple paper dicts."""
    papers = []
    for item in results:
        pid = item.get("id")
        if not pid:
            continue
        title = item.get("title", {})
        title_text = title.get("text", "Unknown") if isinstance(title, dict) else str(title)
        authors_raw = item.get("authors", [])
        author_list = []
        for a in authors_raw:
            if isinstance(a, list) and len(a) > 0:
                author_list.append({"name": a[0].get("name", "")})
            elif isinstance(a, dict):
                author_list.append({"name": a.get("name", "")})
        venue = item.get("venue")
        venue_text = venue.get("text", "") if isinstance(venue, dict) else str(venue or "")
        abstract = item.get("paperAbstract")
        abstract_text = abstract.get("text", "") if isinstance(abstract, dict) else str(abstract or "")
        papers.append({
            "paperId": pid,
            "title": title_text,
            "authors": author_list,
            "year": item.get("year"),
            "citationCount": item.get("numCitedBy", 0),
            "referenceCount": item.get("numCiting", 0),
            "venue": venue_text,
            "abstract": abstract_text,
            "fieldsOfStudy": item.get("fieldsOfStudy", []),
            "isKey": item.get("isKey", False),
        })
    return papers


async def _ensure_s2_cookies(client: httpx.AsyncClient) -> None:
    """Ensure S2 cookies are available. Bootstraps Playwright browser if needed."""
    if _s2_cookies:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_pw_executor, _get_pw_context)


async def _fetch_neighbors_httpx(
    paper_id: str, direction: str, sort: str, page_num: int, limit: int,
    client: httpx.AsyncClient,
) -> dict:
    """Fetch citations/references via httpx + cookies (hierarchy #1)."""
    citation_type = "citingPapers" if direction == "citations" else "citedPapers"
    url = f"{INTERNAL_BASE}/search/paper/{paper_id}/citations"
    payload = {
        "page": page_num, "pageSize": limit, "sort": sort,
        "authors": [], "coAuthors": [], "venues": [],
        "yearFilter": None, "requireViewablePdf": False,
        "fieldsOfStudy": [], "citationType": citation_type,
    }
    await _ensure_s2_cookies(client)
    for attempt in range(5):
        try:
            headers = {**_BROWSER_HEADERS, "x-s2-client": "webapp-browser"}
            response = await client.post(
                url, json=payload, headers=headers, cookies=_s2_cookies,
            )
            if response.status_code == 200:
                data = response.json()
                return {"ok": True, "results": data.get("results", []),
                        "totalResults": data.get("totalResults", 0)}
            if response.status_code == 202:
                await asyncio.sleep(3)
                continue
            if response.status_code == 403:
                _s2_cookies.clear()
                await _ensure_s2_cookies(client)
                await asyncio.sleep(1)
                continue
            return {"ok": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            if attempt == 4:
                return {"ok": False, "error": str(e)}
            await asyncio.sleep(1)
    return {"ok": False, "error": "Still 202 after retries"}


async def fetch_sorted_neighbors(
    paper_id: str, direction: str, sort: str, limit: int, client: httpx.AsyncClient
) -> list[dict] | str:
    """Fetch citations/references via internal API with cookies (hierarchy #1).
    Tracks pagination so repeated calls on the same node fetch the next page.

    When the API caps at 10K results (unreliable sort for popular papers),
    fetches a larger batch and re-sorts client-side by citation count.
    """
    api_sort = SORT_MAP.get(sort, "relevance")
    page_key = (paper_id, direction, api_sort)
    page_num = _expand_pages.get(page_key, 1)

    # Try httpx + cookies first (hierarchy #1)
    result = await _fetch_neighbors_httpx(
        paper_id, direction, api_sort, page_num, limit, client,
    )

    if not isinstance(result, dict) or not result.get("ok"):
        # Fallback to Playwright in-browser fetch (hierarchy #3)
        print("[cite-graph] httpx failed, falling back to Playwright", file=sys.stderr)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _pw_executor, _pw_fetch_neighbors, paper_id, direction, api_sort, page_num, limit
        )

    if not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
        return f"Internal API error: {error}"

    total_results = result.get("totalResults", 0)
    papers = _parse_internal_results(result.get("results", []))

    # When results are capped at 10K and sorting by citation count,
    # the API only sorts within a recent-papers window (unreliable).
    # Re-sort client-side to ensure correct order within the batch.
    if api_sort == "total-citations":
        papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)
    elif api_sort == "pub-date":
        papers.sort(key=lambda p: p.get("year") or 0, reverse=True)

    existing_ids = set(graph_state["nodes"].keys())
    papers = [p for p in papers if p["paperId"] not in existing_ids]
    # Advance page for next call
    _expand_pages[page_key] = page_num + 1
    return papers[:limit]


def _pw_fetch_neighbors(paper_id: str, direction: str, sort: str, page_num: int, limit: int) -> dict:
    """Fallback: fetch citations/references via Playwright in-browser fetch."""
    page = _get_fetch_page()
    citation_type = "citingPapers" if direction == "citations" else "citedPapers"
    js = """
    async (args) => {
        const {paperId, citationType, sort, page, pageSize} = args;
        const url = "https://www.semanticscholar.org/api/1/search/paper/" + paperId + "/citations";
        const payload = {
            page: page, pageSize: pageSize, sort: sort,
            authors: [], coAuthors: [], venues: [],
            yearFilter: null, requireViewablePdf: false,
            fieldsOfStudy: [], citationType: citationType,
        };
        for (let attempt = 0; attempt < 5; attempt++) {
            const resp = await fetch(url, {
                method: "POST",
                headers: {"Content-Type": "application/json", "Accept": "application/json"},
                body: JSON.stringify(payload),
            });
            if (resp.status === 200) {
                const data = await resp.json();
                return {ok: true, results: data.results || [], totalResults: data.totalResults};
            }
            if (resp.status === 202) {
                await new Promise(r => setTimeout(r, 3000));
                continue;
            }
            return {ok: false, error: "HTTP " + resp.status};
        }
        return {ok: false, error: "Still 202 after retries"};
    }
    """
    args = {
        "paperId": paper_id, "citationType": citation_type,
        "sort": sort, "page": page_num, "pageSize": limit,
    }
    try:
        return page.evaluate(js, args)
    except Exception as e:
        print(f"[cite-graph] Playwright fetch_neighbors error: {e}", file=sys.stderr)
        return {"ok": False, "error": str(e)}


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
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                result = await search_papers(query, limit, client)
            if isinstance(result, str):
                return web.json_response({"error": result})
            return web.json_response({"data": result})
        except Exception as e:
            print(f"[cite-graph] Web search error: {e}", file=sys.stderr)
            return web.json_response({"error": f"Search error: {e}"})

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

    async def handle_api_delete(request):
        """API endpoint to delete nodes from the graph and reset related pagination."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)
        node_ids = body.get("node_ids", [])
        if not node_ids:
            return web.json_response({"success": False, "error": "Missing node_ids"})

        deleted = []
        for nid in node_ids:
            if nid in graph_state["nodes"]:
                del graph_state["nodes"][nid]
                deleted.append(nid)
        graph_state["edges"] = [
            e for e in graph_state["edges"]
            if e["source_id"] not in deleted and e["target_id"] not in deleted
        ]
        # Reset all pagination — deletion invalidates page cursors
        _expand_pages.clear()

        return web.json_response({"success": True, "deleted": len(deleted), "graph": get_graph_data()})

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
        app.router.add_post("/api/delete", handle_api_delete)
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
            description="Get papers that CITE the specified paper (newer papers that reference this one). Supports server-side sorting. Optionally add them to the graph.",
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
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "citation-count", "year", "influential"],
                        "description": "Sort order: relevance (default), citation-count, year (recency), influential",
                        "default": "relevance"
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
            description="Get papers that the specified paper REFERENCES (older papers cited by this one). Supports server-side sorting. Optionally add them to the graph.",
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
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "citation-count", "year", "influential"],
                        "description": "Sort order: relevance (default), citation-count, year (recency), influential",
                        "default": "relevance"
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

        elif name in ("get_citations", "get_references"):
            paper_id = arguments.get("paper_id", "")
            limit = min(max(arguments.get("limit", 10), 1), 100)
            sort = arguments.get("sort", "relevance")
            add_to_graph = arguments.get("add_to_graph", False)
            is_citation = name == "get_citations"
            direction = "citations" if is_citation else "references"

            papers = await fetch_sorted_neighbors(paper_id, direction, sort, limit, client)

            if isinstance(papers, str):
                return [TextContent(type="text", text=papers)]

            if not papers:
                return [TextContent(type="text", text=f"No {direction} found for paper: {paper_id}")]

            sort_label = {"relevance": "Relevance", "citation-count": "Citation Count", "year": "Recency", "influential": "Most Influential"}.get(sort, sort)
            result = f"Found {len(papers)} {direction} (sorted by {sort_label}):\n\n"
            added_count = 0

            for i, p in enumerate(papers, 1):
                authors = ", ".join([a.get("name", "") for a in p.get("authors", [])][:2])
                if len(p.get("authors", [])) > 2:
                    authors += " et al."

                result += f"{i}. **{p.get('title', 'Unknown')}**\n"
                result += f"   - Paper ID: `{p.get('paperId')}`\n"
                result += f"   - Authors: {authors}\n"
                result += f"   - Year: {p.get('year', 'N/A')}\n"
                result += f"   - Citations: {p.get('citationCount', 0)}\n\n"

                if add_to_graph and p.get("paperId"):
                    full_paper = await fetch_paper(p["paperId"], client)
                    if full_paper:
                        add_node_to_graph(full_paper, paper_id, is_citation=is_citation)
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
            _expand_pages.clear()
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
