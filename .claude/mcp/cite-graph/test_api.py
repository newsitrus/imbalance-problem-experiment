#!/usr/bin/env python3
"""
Test script for Citation Graph MCP Server
Tests the Semantic Scholar API functions directly.
"""

import asyncio
import httpx

API_BASE = "https://api.semanticscholar.org/graph/v1"
PAPER_FIELDS = "paperId,title,abstract,authors,year,citationCount,venue,url,references,citations"
SEARCH_FIELDS = "paperId,title,authors,year,citationCount,venue"


async def test_search():
    """Test paper search."""
    print("=" * 60)
    print("Testing: Search Papers")
    print("=" * 60)

    query = "attention is all you need"
    url = f"{API_BASE}/paper/search?query={query}&limit=5&fields={SEARCH_FIELDS}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)

        if response.status_code == 200:
            data = response.json()
            papers = data.get("data", [])
            print(f"Found {len(papers)} papers for '{query}':\n")

            for i, paper in enumerate(papers, 1):
                print(f"{i}. {paper.get('title')}")
                print(f"   ID: {paper.get('paperId')}")
                print(f"   Year: {paper.get('year')}")
                print(f"   Citations: {paper.get('citationCount')}")
                print()

            return papers[0].get("paperId") if papers else None
        else:
            print(f"Error: {response.status_code}")
            return None


async def test_get_paper(paper_id: str):
    """Test getting paper details."""
    print("=" * 60)
    print(f"Testing: Get Paper Details ({paper_id[:20]}...)")
    print("=" * 60)

    url = f"{API_BASE}/paper/{paper_id}?fields={PAPER_FIELDS}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)

        if response.status_code == 200:
            paper = response.json()
            print(f"Title: {paper.get('title')}")
            print(f"Year: {paper.get('year')}")
            print(f"Venue: {paper.get('venue')}")
            print(f"Citation Count: {paper.get('citationCount')}")

            authors = [a.get("name") for a in paper.get("authors", [])]
            print(f"Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")

            citations = paper.get("citations") or []
            references = paper.get("references") or []
            print(f"Citations available: {len(citations)}")
            print(f"References available: {len(references)}")
            print()
            return paper
        else:
            print(f"Error: {response.status_code}")
            return None


async def test_get_citations(paper_id: str):
    """Test getting paper citations."""
    print("=" * 60)
    print(f"Testing: Get Citations")
    print("=" * 60)

    url = f"{API_BASE}/paper/{paper_id}/citations?fields={SEARCH_FIELDS}&limit=5"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)

        if response.status_code == 200:
            data = response.json()
            citations = data.get("data", [])
            print(f"Found {len(citations)} citations (papers that cite this paper):\n")

            for i, item in enumerate(citations, 1):
                citing = item.get("citingPaper", {})
                print(f"{i}. {citing.get('title')}")
                print(f"   Year: {citing.get('year')}")
                print()
        else:
            print(f"Error: {response.status_code}")


async def test_get_references(paper_id: str):
    """Test getting paper references."""
    print("=" * 60)
    print(f"Testing: Get References")
    print("=" * 60)

    url = f"{API_BASE}/paper/{paper_id}/references?fields={SEARCH_FIELDS}&limit=5"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)

        if response.status_code == 200:
            data = response.json()
            references = data.get("data", [])
            print(f"Found {len(references)} references (papers this paper cites):\n")

            for i, item in enumerate(references, 1):
                cited = item.get("citedPaper", {})
                print(f"{i}. {cited.get('title')}")
                print(f"   Year: {cited.get('year')}")
                print()
        else:
            print(f"Error: {response.status_code}")


async def main():
    print("\n🔬 Citation Graph MCP Server - API Test\n")

    # Test search
    paper_id = await test_search()

    if paper_id:
        # Test get paper
        await test_get_paper(paper_id)

        # Test citations
        await test_get_citations(paper_id)

        # Test references
        await test_get_references(paper_id)

    print("=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
