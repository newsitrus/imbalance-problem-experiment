#!/usr/bin/env python3
"""
MinerU Parser MCP Server

Exposes MinerU PDF parsing as an MCP tool.
Always uses backend="pipeline" (CPU-only, no GPU required).

MinerU output structure (fixed by library):
  {tmp_dir}/{pdf_stem}/auto/{pdf_stem}.md
  {tmp_dir}/{pdf_stem}/auto/{pdf_stem}_middle.json
  {tmp_dir}/{pdf_stem}/auto/images/
"""

import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ---------------------------------------------------------------------------
# CLI availability check
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
_mineru_bin = SCRIPT_DIR / ".venv" / "bin" / "mineru"
MINERU_AVAILABLE = _mineru_bin.exists()


def _get_mineru_cmd() -> str:
    """Return path to the mineru executable."""
    if _mineru_bin.exists():
        return str(_mineru_bin)
    raise RuntimeError("mineru CLI not found. Run setup.sh first.")


# ---------------------------------------------------------------------------
# Core parsing logic (synchronous — called via run_in_executor)
# ---------------------------------------------------------------------------

def _run_mineru(
    pdf_path: Path,
    tmp_dir: str,
    language: str,
    table_enable: bool,
    formula_enable: bool,
) -> None:
    """Call mineru CLI. Runs in thread executor to avoid blocking event loop."""
    mineru = _get_mineru_cmd()
    cmd = [
        mineru,
        "-p", str(pdf_path),
        "-o", tmp_dir,
        "-m", "auto",
        "-b", "pipeline",
        "-l", language,
        "-t", str(table_enable),
        "-f", str(formula_enable),
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30000,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mineru CLI failed (exit {proc.returncode}):\n{proc.stderr}"
        )


def _collect_output(
    tmp_dir: str,
    pdf_stem: str,
    output_dir: Optional[Path],
) -> dict:
    """
    Read MinerU output from its fixed path and optionally copy to output_dir.

    MinerU always writes to: {tmp_dir}/{pdf_stem}/auto/
    """
    mineru_out = Path(tmp_dir) / pdf_stem / "auto"

    # Read markdown
    md_file = mineru_out / f"{pdf_stem}.md"
    markdown_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""

    # Extract page count from middle.json
    page_count = 0
    middle_file = mineru_out / f"{pdf_stem}_middle.json"
    if middle_file.exists():
        try:
            middle_data = json.loads(middle_file.read_text(encoding="utf-8"))
            if "pdf_info" in middle_data:
                page_count = len(middle_data["pdf_info"])
        except (json.JSONDecodeError, KeyError):
            pass

    # Collect image filenames
    image_names: List[str] = []
    images_dir = mineru_out / "images"
    if images_dir.exists():
        for img_file in sorted(images_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                image_names.append(img_file.name)

    # Optionally copy output to caller-specified directory
    saved_output_dir: Optional[str] = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if md_file.exists():
            shutil.copy2(md_file, output_dir / "content.md")

        if images_dir.exists() and image_names:
            images_dest = output_dir / "images"
            images_dest.mkdir(exist_ok=True)
            for img_name in image_names:
                shutil.copy2(images_dir / img_name, images_dest / img_name)

        saved_output_dir = str(output_dir)

    return {
        "markdown": markdown_content,
        "page_count": page_count,
        "images": image_names,
        "output_dir": saved_output_dir,
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_server() -> Server:
    server = Server("mineru-parser")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="mineru_parse",
                description=(
                    "Parse a PDF file using MinerU (CPU-only pipeline backend). "
                    "Extracts text as Markdown, detects tables and formulas, "
                    "and collects embedded images. "
                    "Markdown content is always returned inline in the response. "
                    "If output_dir is provided, also saves content.md and images/ there. "
                    "WARNING: First call downloads MinerU model weights (~2-5GB, cached). "
                    "Large PDFs may take several minutes on CPU."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string",
                            "description": "Absolute path to the PDF file to parse.",
                        },
                        "output_dir": {
                            "type": ["string", "null"],
                            "description": (
                                "Optional absolute path to save output files "
                                "(content.md and images/ subdirectory). "
                                "If null, content is returned inline only."
                            ),
                            "default": None,
                        },
                        "language": {
                            "type": "string",
                            "description": "OCR language hint: 'en', 'ch', 'korean', 'japan', etc. Default: 'en'.",
                            "default": "en",
                        },
                        "table_enable": {
                            "type": "boolean",
                            "description": "Enable table detection. Default: true.",
                            "default": True,
                        },
                        "formula_enable": {
                            "type": "boolean",
                            "description": "Enable LaTeX formula detection. Default: true.",
                            "default": True,
                        },
                    },
                    "required": ["pdf_path"],
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        if name != "mineru_parse":
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Unknown tool: {name}",
            }))]

        if not MINERU_AVAILABLE:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "mineru CLI not found. Run setup.sh first.",
            }))]

        # Extract arguments
        pdf_path_str = arguments.get("pdf_path", "")
        output_dir_str = arguments.get("output_dir")
        language = arguments.get("language", "en")
        table_enable = arguments.get("table_enable", True)
        formula_enable = arguments.get("formula_enable", True)

        # Pre-flight validation
        if not pdf_path_str:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": "pdf_path is required.",
            }))]

        pdf_path = Path(pdf_path_str)
        if not pdf_path.is_absolute():
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"pdf_path must be absolute, got: {pdf_path_str}",
            }))]
        if not pdf_path.exists():
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"PDF not found: {pdf_path_str}",
            }))]
        if pdf_path.suffix.lower() != ".pdf":
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": f"Expected a .pdf file, got: {pdf_path.suffix}",
            }))]

        output_dir: Optional[Path] = Path(output_dir_str) if output_dir_str else None

        # Run MinerU CLI in a thread to avoid blocking the MCP event loop
        loop = asyncio.get_event_loop()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                await loop.run_in_executor(
                    None,
                    _run_mineru,
                    pdf_path,
                    tmp_dir,
                    language,
                    table_enable,
                    formula_enable,
                )
                result = _collect_output(tmp_dir, pdf_path.stem, output_dir)

        except Exception as exc:
            print(f"MinerU parse error: {exc}", file=sys.stderr)
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": str(exc),
            }))]

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "markdown": result["markdown"],
            "page_count": result["page_count"],
            "images": result["images"],
            "output_dir": result["output_dir"],
        }))]

    return server


async def main() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
