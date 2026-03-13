#!/usr/bin/env python3
"""
Test script for the Citation Graph web visualization server.
Run this directly to test the web interface without MCP.
"""

import asyncio
import json
import threading
import time
import webbrowser
from pathlib import Path
import httpx
import os
from urllib.parse import quote

# Semantic Scholar API
API_BASE = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,url,references,citations"
SEARCH_FIELDS = "paperId,title,authors,year,citationCount,venue"
EXPAND_FIELDS = "paperId,title,authors,year,citationCount,venue,isInfluential"
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

def get_api_headers():
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers

# Add some test data with full abstracts
test_graph = {
    "nodes": {
        "204e3073870fae3d05bcbc2f6a8e263d9b72e776": {
            "id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "title": "Attention Is All You Need",
            "authors": "Vaswani, Shazeer, Parmar et al.",
            "year": 2017,
            "citationCount": 168861,
            "venue": "NeurIPS",
            "url": "https://arxiv.org/abs/1706.03762",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
        },
        "df2b0e26d0599ce3e70df8a9da02e51594e0e992": {
            "id": "df2b0e26d0599ce3e70df8a9da02e51594e0e992",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": "Devlin, Chang, Lee, Toutanova",
            "year": 2019,
            "citationCount": 95000,
            "venue": "NAACL",
            "url": "https://arxiv.org/abs/1810.04805",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers."
        },
        "6b85b63579a916f705a8e10a49bd8d849d91b1fc": {
            "id": "6b85b63579a916f705a8e10a49bd8d849d91b1fc",
            "title": "Language Models are Few-Shot Learners",
            "authors": "Brown, Mann, Ryder et al.",
            "year": 2020,
            "citationCount": 45000,
            "venue": "NeurIPS",
            "url": "https://arxiv.org/abs/2005.14165",
            "abstract": "We show that scaling up language models greatly improves task-agnostic, few-shot performance. We train GPT-3, an autoregressive language model with 175 billion parameters, and test its performance in the few-shot setting."
        },
        "2c03df8b48bf3fa39054345bafabfeff15bfd11d": {
            "id": "2c03df8b48bf3fa39054345bafabfeff15bfd11d",
            "title": "Deep Residual Learning for Image Recognition",
            "authors": "He, Zhang, Ren, Sun",
            "year": 2016,
            "citationCount": 180000,
            "venue": "CVPR",
            "url": "https://arxiv.org/abs/1512.03385",
            "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs."
        },
        "abd1c342495432171beb7ca8fd9551ef13cbd0ff": {
            "id": "abd1c342495432171beb7ca8fd9551ef13cbd0ff",
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "authors": "Krizhevsky, Sutskever, Hinton",
            "year": 2012,
            "citationCount": 120000,
            "venue": "NeurIPS",
            "url": "https://papers.nips.cc/paper/4824-imagenet-classification",
            "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes."
        },
    },
    "edges": [
        {"source_id": "df2b0e26d0599ce3e70df8a9da02e51594e0e992", "target_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776"},
        {"source_id": "6b85b63579a916f705a8e10a49bd8d849d91b1fc", "target_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776"},
        {"source_id": "6b85b63579a916f705a8e10a49bd8d849d91b1fc", "target_id": "df2b0e26d0599ce3e70df8a9da02e51594e0e992"},
        {"source_id": "2c03df8b48bf3fa39054345bafabfeff15bfd11d", "target_id": "abd1c342495432171beb7ca8fd9551ef13cbd0ff"},
        {"source_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776", "target_id": "2c03df8b48bf3fa39054345bafabfeff15bfd11d"},
    ]
}


async def run_test_server():
    """Run test web server with sample data."""
    from aiohttp import web
    import aiohttp

    WEB_HOST = "0.0.0.0"
    WEB_PORT = 8765
    WEB_DIR = Path(__file__).parent / "web"
    clients = set()

    async def handle_index(request):
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="Web interface not found", status=404)

    async def handle_api_graph(request):
        return web.json_response({
            "nodes": list(test_graph["nodes"].values()),
            "edges": test_graph["edges"]
        })

    async def handle_api_search(request):
        """Search papers using Semantic Scholar API."""
        query = request.query.get('query', '')
        try:
            limit = min(int(request.query.get('limit', 10)), 50)
        except (ValueError, TypeError):
            limit = 10

        if not query:
            return web.json_response({"error": "No query provided", "data": []})

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{API_BASE}/paper/search?query={query}&limit={limit}&fields={SEARCH_FIELDS}"
                max_retries = 4
                for attempt in range(max_retries):
                    response = await client.get(url, headers=get_api_headers())

                    if response.status_code == 200:
                        data = response.json()
                        return web.json_response({"data": data.get("data", [])})
                    elif response.status_code == 429:
                        wait = 2 ** (attempt + 1)
                        print(f"Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        print(f"API error: {response.status_code}")
                        return web.json_response({"error": f"Semantic Scholar returned an error. Please try again.", "data": []})

                return web.json_response({"error": "Semantic Scholar is busy. Please wait a moment and try again.", "data": []})
        except Exception as e:
            print(f"Search error: {e}")
            return web.json_response({"error": str(e), "data": []})

    async def handle_api_add_paper(request):
        """Add a paper to the graph."""
        try:
            body = await request.json()
            paper_id = body.get('paper_id')

            if not paper_id:
                return web.json_response({"success": False, "error": "No paper_id provided"})

            # Check if already in graph
            if paper_id in test_graph["nodes"]:
                return web.json_response({"success": True, "message": "Paper already in graph"})

            # Fetch paper details from Semantic Scholar
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{API_BASE}/paper/{paper_id}?fields={PAPER_FIELDS}"
                response = await client.get(url, headers=get_api_headers())

                if response.status_code != 200:
                    return web.json_response({"success": False, "error": "Paper not found"})

                paper = response.json()

            # Add to graph
            test_graph["nodes"][paper_id] = {
                "id": paper_id,
                "title": paper.get("title") or "Unknown",
                "authors": ", ".join([a.get("name", "") for a in (paper.get("authors") or [])]),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount") or 0,
                "venue": paper.get("venue") or "",
                "url": paper.get("url") or "",
                "abstract": paper.get("abstract") or "",
            }

            # Auto-discover edges to existing nodes
            for citation in paper.get("citations", []) or []:
                cit_id = citation.get("paperId")
                if cit_id and cit_id in test_graph["nodes"]:
                    edge = {"source_id": cit_id, "target_id": paper_id}
                    if edge not in test_graph["edges"]:
                        test_graph["edges"].append(edge)

            for reference in paper.get("references", []) or []:
                ref_id = reference.get("paperId")
                if ref_id and ref_id in test_graph["nodes"]:
                    edge = {"source_id": paper_id, "target_id": ref_id}
                    if edge not in test_graph["edges"]:
                        test_graph["edges"].append(edge)

            # Notify WebSocket clients
            for ws in list(clients):
                try:
                    await ws.send_json({
                        "type": "graph_update",
                        "graph": {
                            "nodes": list(test_graph["nodes"].values()),
                            "edges": test_graph["edges"]
                        }
                    })
                except Exception:
                    clients.discard(ws)

            print(f"Added paper: {paper.get('title', paper_id)}")
            return web.json_response({"success": True, "paperId": paper_id})

        except Exception as e:
            print(f"Add paper error: {e}")
            return web.json_response({"success": False, "error": str(e)})

    # --- Internal Semantic Scholar Web API (supports server-side sorting) ---
    # The SS website uses a GET endpoint at /api/1/paper/{id}/citations that
    # supports sort parameters the public graph API does not.
    SS_WEB_BASE = "https://www.semanticscholar.org/api/1"
    SS_WEB_HEADERS = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    # For citations (citingPapers): total-citations sort is broken (returns newest),
    # so we use is-influential and re-sort client-side by numCitedBy.
    CITATION_SORT_MAP = {
        "citation-count": "is-influential",  # fetch influential, re-sort by numCitedBy
        "year": "pub-date",
        "influential": "is-influential",
        "relevance": "relevance",
    }
    REFERENCE_SORT_MAP = {
        "citation-count": "relevance",
        "year": "year",
        "influential": "is-influential",
        "relevance": "relevance",
    }

    async def fetch_sorted_ids_internal(client, paper_id, direction, sort, total_needed):
        """Fetch paper IDs using SS internal web API with server-side sorting.

        Uses GET /api/1/paper/{id}/citations with citationType and sort params.
        Returns list of (paper_id, numCitedBy) tuples, or None if unavailable.
        """
        fetch_size = min(max(total_needed * 10, 100), 100)
        cite_type = "citingPapers" if direction == "citations" else "citedPapers"

        sort_map = CITATION_SORT_MAP if direction == "citations" else REFERENCE_SORT_MAP
        ss_sort = sort_map.get(sort, "is-influential")

        try:
            url = (
                f"{SS_WEB_BASE}/paper/{paper_id}/citations"
                f"?sort={ss_sort}&citationType={cite_type}&citationsPageSize={fetch_size}"
            )
            resp = await client.get(url, headers=SS_WEB_HEADERS)
            if resp.status_code != 200:
                print(f"Internal API returned {resp.status_code}")
                return None

            data = resp.json()
            results = data.get("citations") or []

            # Extract IDs with citation counts for client-side re-sorting
            papers = []
            for r in results:
                pid = r.get("id")
                if pid:
                    papers.append((str(pid), r.get("numCitedBy", 0)))

            # For citation-count sort, re-sort by numCitedBy descending
            if sort == "citation-count":
                papers.sort(key=lambda x: x[1], reverse=True)

            print(f"Internal API: fetched {len(papers)} sorted {direction} IDs (sort={ss_sort})")
            if papers:
                top = papers[:3]
                for pid, cc in top:
                    print(f"  top: {pid[:16]}... cited_by={cc}")

            return [pid for pid, _ in papers]

        except Exception as e:
            print(f"Internal API error: {e}")
            return None

    async def fetch_sorted_ids_public(client, paper_id, direction, sort, total_needed):
        """Fallback: fetch from public graph API, sort client-side."""
        encoded_id = quote(paper_id, safe=":")
        url = f"{API_BASE}/paper/{encoded_id}/{direction}"
        params = {"fields": EXPAND_FIELDS, "limit": min(total_needed * 5, 500)}

        max_retries = 4
        for attempt in range(max_retries):
            resp = await client.get(url, params=params, headers=get_api_headers())
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"Public API rate limited, retry in {wait}s ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"Public API returned {resp.status_code}")
                return None, resp.status_code
        else:
            return None, 429

        data = resp.json()
        items = data.get("data", [])
        key = "citingPaper" if direction == "citations" else "citedPaper"

        papers = []
        for item in items:
            p = item.get(key, {})
            if not p or not p.get("paperId"):
                continue
            p["isInfluential"] = item.get("isInfluential", False)
            papers.append(p)

        # Client-side sort
        if sort == "citation-count":
            papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        elif sort == "year":
            papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
        elif sort == "influential":
            papers.sort(key=lambda p: (p.get("isInfluential", False), p.get("citationCount") or 0), reverse=True)

        return [p["paperId"] for p in papers], 200

    async def fetch_paper_details_with_retry(client, pid, max_retries=4):
        """Fetch full paper details with exponential backoff retry."""
        full_url = f"{API_BASE}/paper/{quote(pid, safe=':')}?fields={PAPER_FIELDS}"
        for attempt in range(max_retries):
            resp = await client.get(full_url, headers=get_api_headers())
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited fetching {pid[:12]}..., retry in {wait}s ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"  Failed to fetch {pid[:12]}...: HTTP {resp.status_code}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    return None
        return None

    async def handle_api_expand(request):
        """Expand a node with citations or references."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid request body"}, status=400)

        paper_id = body.get("paper_id", "")
        direction = body.get("direction", "citations")
        sort = body.get("sort", "citation-count")
        limit = min(int(body.get("limit", 5)), 20)

        if not paper_id:
            return web.json_response({"success": False, "error": "Missing paper_id"}, status=400)
        if direction not in ("citations", "references"):
            return web.json_response({"success": False, "error": "direction must be 'citations' or 'references'"}, status=400)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Step 1: Try internal SS web API (supports server-side sorting)
                paper_ids = await fetch_sorted_ids_internal(client, paper_id, direction, sort, limit)

                if paper_ids is not None:
                    print(f"Using internal API for sorted {direction}")
                else:
                    # Fallback to public API with client-side sorting
                    print(f"Internal API unavailable, falling back to public API")
                    result, status = await fetch_sorted_ids_public(client, paper_id, direction, sort, limit)
                    if result is None:
                        if status == 404:
                            return web.json_response({"success": False, "error": "Paper not found on Semantic Scholar."})
                        elif status == 429:
                            return web.json_response({"success": False, "error": "Rate limited. Please wait and try again."})
                        else:
                            return web.json_response({"success": False, "error": f"Semantic Scholar error (HTTP {status})."})
                    paper_ids = result

                # Step 2: Filter out papers already in graph
                existing_ids = set(test_graph["nodes"].keys())
                paper_ids = [pid for pid in paper_ids if pid not in existing_ids]

                if not paper_ids:
                    return web.json_response({
                        "success": True, "added": [], "direction": direction, "sort": sort,
                        "graph": {"nodes": list(test_graph["nodes"].values()), "edges": test_graph["edges"]},
                        "message": f"No new {direction} found",
                    })

                # Step 3: Fetch full details for top candidates until we reach limit
                added = []
                for pid in paper_ids:
                    if len(added) >= limit:
                        break

                    paper = await fetch_paper_details_with_retry(client, pid)
                    if paper is None:
                        continue

                    pid = paper["paperId"]
                    test_graph["nodes"][pid] = {
                        "id": pid,
                        "title": paper.get("title") or "Unknown",
                        "authors": ", ".join([a.get("name", "") for a in (paper.get("authors") or [])]),
                        "year": paper.get("year"),
                        "citationCount": paper.get("citationCount") or 0,
                        "venue": paper.get("venue") or "",
                        "url": paper.get("url") or "",
                        "abstract": paper.get("abstract") or "",
                    }

                    # Add edge
                    if direction == "citations":
                        edge = {"source_id": pid, "target_id": paper_id}
                    else:
                        edge = {"source_id": paper_id, "target_id": pid}
                    if edge not in test_graph["edges"]:
                        test_graph["edges"].append(edge)

                    # Auto-discover connections
                    for cit in paper.get("citations", []) or []:
                        cid = cit.get("paperId")
                        if cid and cid in test_graph["nodes"]:
                            e = {"source_id": cid, "target_id": pid}
                            if e not in test_graph["edges"]:
                                test_graph["edges"].append(e)
                    for ref in paper.get("references", []) or []:
                        rid = ref.get("paperId")
                        if rid and rid in test_graph["nodes"]:
                            e = {"source_id": pid, "target_id": rid}
                            if e not in test_graph["edges"]:
                                test_graph["edges"].append(e)

                    added.append({
                        "paperId": pid,
                        "title": paper.get("title", "Unknown"),
                        "year": paper.get("year"),
                        "citationCount": paper.get("citationCount", 0),
                    })

                # Notify WebSocket clients
                graph_data = {"nodes": list(test_graph["nodes"].values()), "edges": test_graph["edges"]}
                for ws in list(clients):
                    try:
                        await ws.send_json({"type": "graph_update", "graph": graph_data})
                    except Exception:
                        clients.discard(ws)

                print(f"Expanded {paper_id}: added {len(added)} {direction}")
                for a in added:
                    print(f"  + {a['title'][:60]} (citations: {a['citationCount']}, year: {a['year']})")
                return web.json_response({
                    "success": True, "added": added, "direction": direction, "sort": sort,
                    "graph": graph_data,
                })

        except Exception as e:
            print(f"Expand error: {e}")
            return web.json_response({"success": False, "error": str(e)})

    async def handle_api_delete(request):
        """Delete nodes from the graph."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        node_ids = body.get("node_ids", [])
        if not node_ids:
            return web.json_response({"success": False, "error": "No node_ids provided"}, status=400)

        ids_to_delete = set(node_ids)
        removed = 0
        for nid in ids_to_delete:
            if nid in test_graph["nodes"]:
                del test_graph["nodes"][nid]
                removed += 1

        test_graph["edges"] = [
            e for e in test_graph["edges"]
            if e["source_id"] not in ids_to_delete and e["target_id"] not in ids_to_delete
        ]

        graph_data = {"nodes": list(test_graph["nodes"].values()), "edges": test_graph["edges"]}

        # Notify WebSocket clients
        for ws in list(clients):
            try:
                await ws.send_json({"type": "graph_update", "graph": graph_data})
            except Exception:
                clients.discard(ws)

        print(f"Deleted {removed} node(s), {len(test_graph['nodes'])} remaining")
        return web.json_response({"success": True, "removed": removed, "graph": graph_data})

    async def handle_api_import(request):
        """Import a graph from JSON."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)

        nodes = body.get("nodes")
        edges = body.get("edges")

        if not isinstance(nodes, list) or not isinstance(edges, list):
            return web.json_response({"success": False, "error": "Invalid format: need nodes and edges arrays"}, status=400)

        # Replace graph state
        test_graph["nodes"] = {}
        test_graph["edges"] = []

        for n in nodes:
            nid = n.get("id")
            if not nid:
                continue
            test_graph["nodes"][nid] = {
                "id": nid,
                "title": n.get("title") or "Unknown",
                "authors": n.get("authors") or "",
                "year": n.get("year"),
                "citationCount": n.get("citationCount") or 0,
                "venue": n.get("venue") or "",
                "url": n.get("url") or "",
                "abstract": n.get("abstract") or "",
            }

        for e in edges:
            sid = e.get("source_id") or (e.get("source", {}).get("id") if isinstance(e.get("source"), dict) else e.get("source"))
            tid = e.get("target_id") or (e.get("target", {}).get("id") if isinstance(e.get("target"), dict) else e.get("target"))
            if sid and tid:
                test_graph["edges"].append({"source_id": sid, "target_id": tid})

        graph_data = {"nodes": list(test_graph["nodes"].values()), "edges": test_graph["edges"]}

        # Notify WebSocket clients
        for ws in list(clients):
            try:
                await ws.send_json({"type": "graph_update", "graph": graph_data})
            except Exception:
                clients.discard(ws)

        print(f"Imported graph: {len(test_graph['nodes'])} nodes, {len(test_graph['edges'])} edges")
        return web.json_response({"success": True, "graph": graph_data})

    async def handle_websocket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        print(f"WebSocket client connected. Total: {len(clients)}")

        try:
            await ws.send_json({
                "type": "graph_update",
                "graph": {
                    "nodes": list(test_graph["nodes"].values()),
                    "edges": test_graph["edges"]
                }
            })

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            clients.discard(ws)
            print(f"WebSocket client disconnected. Total: {len(clients)}")

        return ws

    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/graph", handle_api_graph)
    app.router.add_get("/api/search", handle_api_search)
    app.router.add_post("/api/add_paper", handle_api_add_paper)
    app.router.add_post("/api/expand", handle_api_expand)
    app.router.add_post("/api/delete", handle_api_delete)
    app.router.add_post("/api/import", handle_api_import)
    app.router.add_get("/ws", handle_websocket)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, WEB_HOST, WEB_PORT)
    await site.start()

    url = f"http://{WEB_HOST}:{WEB_PORT}"
    print(f"\n{'='*60}")
    print(f"🌐 Citation Graph Visualization Server")
    print(f"{'='*60}")
    print(f"\nServer running at: {url}")
    print(f"Test data: {len(test_graph['nodes'])} papers, {len(test_graph['edges'])} citations")
    print(f"\nPress Ctrl+C to stop\n")

    # Open browser
    webbrowser.open(url)

    # Keep running
    while True:
        await asyncio.sleep(1)


def main():
    try:
        asyncio.run(run_test_server())
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


if __name__ == "__main__":
    main()
