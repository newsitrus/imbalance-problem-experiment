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

# Semantic Scholar API
API_BASE = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,url,references,citations"
SEARCH_FIELDS = "paperId,title,authors,year,citationCount,venue"
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

def get_api_headers():
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    return headers

# Add some test data with full abstracts
test_graph = {
    "nodes": {
        "paper1": {
            "id": "paper1",
            "title": "Attention Is All You Need",
            "authors": "Vaswani, Shazeer, Parmar et al.",
            "year": 2017,
            "citationCount": 168861,
            "venue": "NeurIPS",
            "url": "https://arxiv.org/abs/1706.03762",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
        },
        "paper2": {
            "id": "paper2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": "Devlin, Chang, Lee, Toutanova",
            "year": 2019,
            "citationCount": 95000,
            "venue": "NAACL",
            "url": "https://arxiv.org/abs/1810.04805",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."
        },
        "paper3": {
            "id": "paper3",
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "authors": "Brown, Mann, Ryder et al.",
            "year": 2020,
            "citationCount": 45000,
            "venue": "NeurIPS",
            "url": "https://arxiv.org/abs/2005.14165",
            "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic."
        },
        "paper4": {
            "id": "paper4",
            "title": "Deep Residual Learning for Image Recognition",
            "authors": "He, Zhang, Ren, Sun",
            "year": 2016,
            "citationCount": 180000,
            "venue": "CVPR",
            "url": "https://arxiv.org/abs/1512.03385",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation."
        },
        "paper5": {
            "id": "paper5",
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "authors": "Krizhevsky, Sutskever, Hinton",
            "year": 2012,
            "citationCount": 120000,
            "venue": "NeurIPS",
            "url": "https://papers.nips.cc/paper/4824-imagenet-classification",
            "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called dropout that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry."
        },
    },
    "edges": [
        {"source_id": "paper2", "target_id": "paper1"},  # BERT cites Attention
        {"source_id": "paper3", "target_id": "paper1"},  # GPT-3 cites Attention
        {"source_id": "paper3", "target_id": "paper2"},  # GPT-3 cites BERT
        {"source_id": "paper4", "target_id": "paper5"},  # ResNet cites AlexNet
        {"source_id": "paper1", "target_id": "paper4"},  # Attention cites ResNet (indirect)
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
        limit = min(int(request.query.get('limit', 10)), 50)

        if not query:
            return web.json_response({"error": "No query provided", "data": []})

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"{API_BASE}/paper/search?query={query}&limit={limit}&fields={SEARCH_FIELDS}"
                response = await client.get(url, headers=get_api_headers())

                if response.status_code == 200:
                    data = response.json()
                    return web.json_response({"data": data.get("data", [])})
                elif response.status_code == 429:
                    print("Rate limited by Semantic Scholar API")
                    return web.json_response({"error": "Rate limited. Please wait a moment and try again.", "data": []})
                else:
                    print(f"API error: {response.status_code}")
                    return web.json_response({"error": f"API error: {response.status_code}", "data": []})
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
                "title": paper.get("title", "Unknown"),
                "authors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount", 0),
                "venue": paper.get("venue", ""),
                "url": paper.get("url", ""),
                "abstract": paper.get("abstract", ""),
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
