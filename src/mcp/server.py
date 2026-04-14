"""MCP (Model Context Protocol) server for Deliberative RAG.

Exposes the pipeline's unique capabilities as MCP tools that any
MCP-compatible AI application (Claude Desktop, Cursor, VS Code Copilot)
can discover and invoke.

Tools exposed:
- ``deliberative_rag/research`` — full deep research with conflict detection
- ``deliberative_rag/conflict_detect`` — detect contradictions between passages
- ``deliberative_rag/claim_extract`` — extract atomic claims from text
- ``deliberative_rag/score_claims`` — score claims by recency + authority
- ``deliberative_rag/web_search`` — search the web for current information

Run with::

    python -m src.mcp.server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

log = structlog.get_logger()

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_mcp_server() -> Server:
    """Build and configure the MCP server with all tools registered.

    Returns:
        Configured MCP Server instance.
    """
    server = Server("deliberative-rag")

    # ------------------------------------------------------------------
    # Tool listing
    # ------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return the list of available tools."""
        return [
            Tool(
                name="deliberative_rag/research",
                description=(
                    "Run a full deep research query with conflict-aware analysis. "
                    "Searches both web and knowledge base, detects contradictions "
                    "between sources, and returns an answer with confidence score "
                    "and reasoning trace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The research question to investigate.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="deliberative_rag/conflict_detect",
                description=(
                    "Detect contradictions between given text passages. "
                    "Extracts claims, builds a conflict graph, and identifies "
                    "CONTRADICTS and SUPERSEDES relationships."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "passages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of text passages to check for conflicts.",
                        },
                    },
                    "required": ["passages"],
                },
            ),
            Tool(
                name="deliberative_rag/claim_extract",
                description=(
                    "Extract atomic factual claims from text passages. "
                    "Each claim is a standalone assertion with a type "
                    "(fact, opinion, forecast, data_point) and confidence score."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract claims from.",
                        },
                    },
                    "required": ["text"],
                },
            ),
            Tool(
                name="deliberative_rag/score_claims",
                description=(
                    "Score claims by temporal recency, source authority, "
                    "and graph evidence. Returns a ranked list with composite scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "source_type": {"type": "string"},
                                    "publication_date": {"type": "string"},
                                },
                            },
                            "description": "Claims to score with metadata.",
                        },
                    },
                    "required": ["claims"],
                },
            ),
            Tool(
                name="deliberative_rag/web_search",
                description=(
                    "Search the web for current information using Tavily. "
                    "Returns results with titles, URLs, content, and publication dates."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of results (1-10, default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="deliberative_rag/vectorless_retrieve",
                description=(
                    "Retrieve relevant passages from uploaded documents using "
                    "vectorless retrieval (BM25 + cross-encoder reranking). "
                    "No vector database or pre-embedding needed — works on "
                    "raw text at query time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "passages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Text passages to search through.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query", "passages"],
                },
            ),
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Execute a tool and return results."""
        try:
            if name == "deliberative_rag/research":
                result = await _handle_research(arguments)
            elif name == "deliberative_rag/conflict_detect":
                result = await _handle_conflict_detect(arguments)
            elif name == "deliberative_rag/claim_extract":
                result = await _handle_claim_extract(arguments)
            elif name == "deliberative_rag/score_claims":
                result = await _handle_score_claims(arguments)
            elif name == "deliberative_rag/web_search":
                result = await _handle_web_search(arguments)
            elif name == "deliberative_rag/vectorless_retrieve":
                result = await _handle_vectorless_retrieve(arguments)
            else:
                result = f"Unknown tool: {name}"

            return [TextContent(type="text", text=result)]

        except Exception as exc:
            log.error("mcp_tool_error", tool=name, error=str(exc))
            return [TextContent(type="text", text=f"Error: {exc}")]

    return server


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def _handle_research(arguments: dict) -> str:
    """Handle the full research tool."""
    from src.agent.deep_research import run_deep_research

    query = arguments.get("query", "")
    if not query:
        return "Error: 'query' is required."

    result = run_deep_research(query)
    return result.get("answer", "No result produced.")


async def _handle_conflict_detect(arguments: dict) -> str:
    """Handle the conflict detection tool."""
    from src.pipeline.claim_extractor import ClaimExtractor
    from src.pipeline.conflict_graph import ConflictGraphBuilder
    from src.utils.llm import get_llm
    from src.vectorstore.embeddings import EmbeddingModel

    passages = arguments.get("passages", [])
    if not passages:
        return "Error: 'passages' list is required."

    llm = get_llm("light")
    embedder = EmbeddingModel()

    # Step 1: Extract claims from all passages
    extractor = ClaimExtractor(llm)
    passage_dicts = [
        {"text": p, "source_doc_id": f"passage_{i}"}
        for i, p in enumerate(passages)
    ]
    claims = extractor.extract_claims_combined(passage_dicts, query="")

    if not claims:
        return "No claims could be extracted from the provided passages."

    # Step 2: Build conflict graph
    builder = ConflictGraphBuilder(llm, embedder)
    graph = builder.build_graph(claims)

    # Step 3: Format results
    conflicts = []
    for u, v, data in graph.edges(data=True):
        edge = data.get("edge")
        if edge and edge.relation.value in ("CONTRADICTS", "SUPERSEDES"):
            conflicts.append({
                "source_claim": edge.source_claim_id,
                "target_claim": edge.target_claim_id,
                "relation": edge.relation.value,
                "confidence": edge.confidence,
                "reasoning": edge.reasoning,
            })

    result = {
        "num_claims": len(claims),
        "num_edges": graph.number_of_edges(),
        "num_conflicts": len(conflicts),
        "claims": [
            {"id": c.claim_id, "text": c.text, "type": c.claim_type.value}
            for c in claims
        ],
        "conflicts": conflicts,
    }

    return json.dumps(result, indent=2)


async def _handle_claim_extract(arguments: dict) -> str:
    """Handle the claim extraction tool."""
    from src.pipeline.claim_extractor import ClaimExtractor
    from src.utils.llm import get_llm

    text = arguments.get("text", "")
    if not text:
        return "Error: 'text' is required."

    llm = get_llm("light")
    extractor = ClaimExtractor(llm)
    claims = extractor.extract_claims(text, {"source_doc_id": "mcp_input"})

    result = [
        {
            "claim_id": c.claim_id,
            "text": c.text,
            "type": c.claim_type.value,
            "temporal_marker": c.temporal_marker,
            "confidence": c.confidence_in_extraction,
        }
        for c in claims
    ]

    return json.dumps(result, indent=2)


async def _handle_score_claims(arguments: dict) -> str:
    """Handle the claim scoring tool."""
    import uuid

    import networkx as nx

    from src.pipeline.scorer import ClaimScorer
    from src.schemas import Claim, ClaimType

    raw_claims = arguments.get("claims", [])
    if not raw_claims:
        return "Error: 'claims' list is required."

    # Build Claim objects from input
    claims = []
    for rc in raw_claims:
        claims.append(Claim(
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            text=rc.get("text", ""),
            claim_type=ClaimType.FACT,
            source_doc_id="mcp_input",
            temporal_marker=rc.get("publication_date"),
            confidence_in_extraction=0.9,
        ))

    # Score with empty graph (no conflict edges available)
    scorer = ClaimScorer()
    graph = nx.DiGraph()
    for c in claims:
        graph.add_node(c.claim_id, claim=c)

    metadata = {}
    for i, rc in enumerate(raw_claims):
        metadata[claims[i].source_doc_id] = {
            "source_type": rc.get("source_type", ""),
            "publication_date": rc.get("publication_date"),
        }

    scored = scorer.score_all_claims(claims, graph, metadata=metadata)

    result = [
        {
            "text": sc.text,
            "temporal_score": sc.temporal_score,
            "authority_score": sc.authority_score,
            "graph_evidence_score": sc.graph_evidence_score,
            "final_score": sc.final_score,
        }
        for sc in scored
    ]

    return json.dumps(result, indent=2)


async def _handle_web_search(arguments: dict) -> str:
    """Handle the web search tool."""
    from src.tools.web_search import web_search as ws

    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)

    if not query:
        return "Error: 'query' is required."

    return ws(query, max_results=max_results)


async def _handle_vectorless_retrieve(arguments: dict) -> str:
    """Handle the vectorless retrieval tool — BM25 + cross-encoder reranking."""
    from src.pipeline.vectorless_retriever import VectorlessRetriever

    query = arguments.get("query", "")
    passages = arguments.get("passages", [])
    top_k = arguments.get("top_k", 5)

    if not query:
        return "Error: 'query' is required."
    if not passages:
        return "Error: 'passages' list is required."

    # Build document dicts from raw passages
    docs = [
        {"id": f"passage_{i}", "text": p, "source": "mcp_input"}
        for i, p in enumerate(passages)
    ]

    retriever = VectorlessRetriever(use_cross_encoder=True)
    retriever.add_documents(docs)

    if len(docs) <= 50:
        results = retriever.retrieve_all_scored(query)[:top_k]
    else:
        results = retriever.retrieve(query, top_k=top_k)

    output = [
        {
            "text": r.get("text", "")[:500],
            "score": round(r.get("score", 0.0), 4),
            "method": r.get("retrieval_method", "unknown"),
        }
        for r in results
    ]

    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the MCP server on stdio transport."""
    server = create_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
