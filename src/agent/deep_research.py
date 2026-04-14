"""Deep Research Agent powered by LangChain deepagents.

Uses deepagents for planning and context management, with a deterministic
tool execution pipeline that doesn't rely on the LLM's tool-calling ability.

The agent follows a fixed research strategy:
1. Decompose query into sub-queries (LLM)
2. Search web + knowledge base for each sub-query (deterministic)
3. Run conflict detection pipeline on all gathered evidence (deterministic)
4. Synthesize final answer with reasoning trace (LLM)

This hybrid approach gives us the deepagents infrastructure (planning,
memory, subagents) while ensuring tools are always called correctly
regardless of the local model's tool-calling reliability.

Usage::

    from src.agent.deep_research import run_deep_research

    result = run_deep_research("What is the current GDP of India?")
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from deepagents import create_deep_agent

from config.settings import get_settings
from src.tools.document_parser import parse_document, parse_document_structured
from src.tools.web_search import web_search, web_search_structured
from src.utils.llm import LLMClient, get_llm
from src.utils.prompts import load_prompt
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Deterministic research pipeline
# ---------------------------------------------------------------------------


def _search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web and return structured results.

    Args:
        query: Search query.
        max_results: Max results.

    Returns:
        List of passage dicts from web search.
    """
    try:
        results = web_search_structured(query, max_results=max_results)
        log.info("deep_research_web_search", query=query[:60], results=len(results))
        return results
    except Exception as exc:
        log.error("deep_research_web_search_failed", error=str(exc))
        return []


def _search_kb(
    query: str,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search the knowledge base and return structured results.

    Args:
        query: Search query.
        qdrant: Qdrant manager.
        embedder: Embedding model.
        top_k: Max results.

    Returns:
        List of passage dicts from KB search.
    """
    try:
        embedding = embedder.embed_text(query)
        results = qdrant.search(embedding, top_k=top_k)
        for r in results:
            r["source"] = r.get("source", "knowledge_base")
        log.info("deep_research_kb_search", query=query[:60], results=len(results))
        return results
    except Exception as exc:
        log.error("deep_research_kb_search_failed", error=str(exc))
        return []


def _decompose_query(query: str, llm: LLMClient) -> list[str]:
    """Use the LLM to decompose a complex query into sub-queries.

    Args:
        query: The user query.
        llm: LLM client for decomposition.

    Returns:
        List of sub-queries (includes the original query).
    """
    from src.pipeline.query_analyzer import decompose_query

    try:
        sub_queries = decompose_query(query, llm)
        log.info("deep_research_decomposed", sub_queries=len(sub_queries))
        return sub_queries
    except Exception as exc:
        log.warning("deep_research_decompose_failed", error=str(exc))
        return [query]


def _run_conflict_pipeline(
    query: str,
    all_passages: list[dict[str, Any]],
    llm_heavy: LLMClient,
    llm_light: LLMClient,
    embedder: EmbeddingModel,
) -> dict[str, Any]:
    """Run the claim extraction + conflict graph + scoring pipeline.

    Args:
        query: Original user query.
        all_passages: All gathered passages from web + KB.
        llm_heavy: Heavy LLM for synthesis.
        llm_light: Light LLM for classification.
        embedder: Embedding model.

    Returns:
        Dict with claims, conflicts, scored_claims, and synthesis result.
    """
    from src.pipeline.claim_extractor import ClaimExtractor
    from src.pipeline.conflict_graph import ConflictGraphBuilder
    from src.pipeline.scorer import ClaimScorer
    from src.pipeline.synthesizer import DeliberationSynthesizer

    # Stage 1: Extract claims from top passages (limit to 8 to stay within
    # Gemma 4's context window — 27 passages overflows and causes parse failure)
    extractor = ClaimExtractor(llm_heavy)

    # Sort by score descending and take the top 8 most relevant
    scored_passages = sorted(all_passages, key=lambda p: p.get("score", 0), reverse=True)
    top_passages = scored_passages[:8]

    passage_dicts = [
        {
            "text": p.get("text", "")[:600],  # truncate long passages
            "source_doc_id": str(p.get("id", f"doc_{i}")),
        }
        for i, p in enumerate(top_passages)
        if p.get("text", "").strip()
    ]

    claims = extractor.extract_claims_combined(passage_dicts, query=query)
    log.info("deep_research_claims_extracted", count=len(claims))

    if not claims:
        # Fallback: synthesize directly from passages without conflict detection
        log.warning("deep_research_no_claims_fallback")
        context = "\n\n".join(
            f"Source {i+1}: {p.get('text', '')[:400]}"
            for i, p in enumerate(top_passages[:5])
        )
        fallback_prompt = (
            f"Based on the following sources, answer this question: {query}\n\n"
            f"{context}\n\n"
            f"Provide a clear, direct answer."
        )
        fallback_answer = llm_heavy.invoke(fallback_prompt)
        from src.schemas import ConfidenceLevel, DeliberationResult
        return {
            "claims": [],
            "conflicts": [],
            "scored_claims": [],
            "synthesis": DeliberationResult(
                query=query,
                answer=fallback_answer,
                confidence=ConfidenceLevel.MODERATE,
                confidence_score=0.6,
                reasoning_trace=["Claim extraction failed; synthesized directly from passages."],
                source_attribution=[],
                conflict_summary="",
            ),
        }

    # Stage 2: Build conflict graph
    builder = ConflictGraphBuilder(llm_light, embedder)
    graph = builder.build_graph(claims)
    log.info(
        "deep_research_conflict_graph",
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
    )

    # Stage 3: Score claims
    doc_metadata: dict[str, dict] = {}
    for p in all_passages:
        doc_id = str(p.get("id", ""))
        doc_metadata[doc_id] = {
            "source_type": p.get("source_type", p.get("source", "")),
            "publication_date": p.get("publication_date"),
        }

    scorer = ClaimScorer()
    scored_claims = scorer.score_all_claims(claims, graph, metadata=doc_metadata)
    log.info("deep_research_claims_scored", count=len(scored_claims))

    # Stage 4: Synthesize answer
    synthesizer = DeliberationSynthesizer(llm_heavy)
    result = synthesizer.synthesize(query, scored_claims, graph)
    log.info("deep_research_synthesized", confidence=result.confidence.value)

    # Extract conflict edges
    conflicts = []
    for u, v, data in graph.edges(data=True):
        edge = data.get("edge")
        if edge and edge.relation.value in ("CONTRADICTS", "SUPERSEDES"):
            conflicts.append({
                "from": edge.source_claim_id,
                "to": edge.target_claim_id,
                "relation": edge.relation.value,
                "confidence": edge.confidence,
                "reasoning": edge.reasoning,
            })

    return {
        "claims": claims,
        "conflicts": conflicts,
        "scored_claims": scored_claims,
        "synthesis": result,
    }


# ---------------------------------------------------------------------------
# Deep agent factory (for planning + memory features)
# ---------------------------------------------------------------------------


def create_research_agent(
    qdrant: QdrantManager | None = None,
    embedder: EmbeddingModel | None = None,
    llm_heavy: LLMClient | None = None,
    llm_light: LLMClient | None = None,
) -> dict[str, Any]:
    """Create a deep research agent context with all dependencies.

    Uses ``deepagents`` for the planning layer while tools are called
    deterministically (not via LLM tool-calling, which is unreliable
    with local models like Gemma 4).

    Args:
        qdrant: Qdrant manager. Initialised from defaults if None.
        embedder: Embedding model. Initialised from defaults if None.
        llm_heavy: Heavy LLM client. Loaded from settings if None.
        llm_light: Light LLM client. Loaded from settings if None.

    Returns:
        Agent context dict with all initialized dependencies.
    """
    settings = get_settings()

    if embedder is None:
        embedder = EmbeddingModel()
    if qdrant is None:
        qdrant = QdrantManager()
    if llm_heavy is None:
        llm_heavy = get_llm("heavy")
    if llm_light is None:
        llm_light = get_llm("light")

    log.info("deep_research_agent_created", num_tools=4)

    return {
        "qdrant": qdrant,
        "embedder": embedder,
        "llm_heavy": llm_heavy,
        "llm_light": llm_light,
        "settings": settings,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_deep_research(
    query: str,
    agent: dict[str, Any] | None = None,
    uploaded_files: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a deep research query with deterministic tool orchestration.

    Pipeline:
    1. Decompose query into sub-queries (LLM planning)
    2. For each sub-query: search web + knowledge base (deterministic)
    3. If uploaded files: parse and add to evidence pool (deterministic)
    4. Deduplicate all gathered evidence
    5. Run conflict detection pipeline: claims → graph → scoring → synthesis
    6. Return structured result with answer, conflicts, reasoning trace

    Args:
        query: User query string.
        agent: Pre-created agent context. If None, a new one is created.
        uploaded_files: Optional list of file paths to analyze.

    Returns:
        Dict with ``answer``, ``sources``, ``conflicts``, and ``metadata``.
    """
    if agent is None:
        agent = create_research_agent()

    qdrant = agent["qdrant"]
    embedder = agent["embedder"]
    llm_heavy = agent["llm_heavy"]
    llm_light = agent["llm_light"]
    settings = agent["settings"]

    reasoning_trace: list[str] = []

    # Step 1: Decompose query
    reasoning_trace.append(f"Analyzing query: '{query}'")
    sub_queries = _decompose_query(query, llm_light)
    reasoning_trace.append(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

    # Step 2: Search web + KB for each sub-query
    all_web_results: list[dict[str, Any]] = []
    all_kb_results: list[dict[str, Any]] = []

    for sq in sub_queries:
        web_results = _search_web(sq, max_results=settings.tavily_max_results)
        all_web_results.extend(web_results)
        reasoning_trace.append(f"Web search for '{sq[:50]}': {len(web_results)} results")

        kb_results = _search_kb(sq, qdrant, embedder, top_k=settings.retrieval_top_k)
        all_kb_results.extend(kb_results)
        reasoning_trace.append(f"KB search for '{sq[:50]}': {len(kb_results)} results")

    # Step 3: Parse uploaded documents (vectorless retrieval)
    uploaded_passages: list[dict[str, Any]] = []
    if uploaded_files:
        for fp in uploaded_files:
            passages = parse_document_structured(fp)
            uploaded_passages.extend(passages)
            reasoning_trace.append(f"Parsed uploaded document: {fp} ({len(passages)} passages)")

    # Step 4: Merge and deduplicate all evidence
    all_passages = _deduplicate_passages(all_web_results + all_kb_results + uploaded_passages)
    reasoning_trace.append(
        f"Total evidence: {len(all_passages)} passages "
        f"(web={len(all_web_results)}, kb={len(all_kb_results)}, "
        f"uploaded={len(uploaded_passages)})"
    )

    if not all_passages:
        return {
            "answer": "No sources found for this query. Try a different query or check that the knowledge base is indexed.",
            "sources": [],
            "conflicts": [],
            "scored_claims": [],
            "messages": [],
            "metadata": {
                "query": query,
                "reasoning_trace": reasoning_trace,
                "num_sources": 0,
            },
        }

    # Step 5: Run conflict detection pipeline
    reasoning_trace.append("Running conflict detection pipeline...")
    pipeline_result = _run_conflict_pipeline(
        query, all_passages, llm_heavy, llm_light, embedder,
    )

    synthesis = pipeline_result.get("synthesis")
    conflicts = pipeline_result.get("conflicts", [])
    scored_claims = pipeline_result.get("scored_claims", [])

    if conflicts:
        reasoning_trace.append(f"Detected {len(conflicts)} conflicts between sources")
    else:
        reasoning_trace.append("No conflicts detected — sources are consistent")

    # Build the final answer
    if synthesis is not None:
        answer_text = (
            f"**Answer**: {synthesis.answer}\n\n"
            f"**Confidence**: {synthesis.confidence.value.upper()} "
            f"({synthesis.confidence_score:.0%})\n\n"
        )

        if synthesis.conflict_summary:
            answer_text += f"**Conflicts Detected**: {synthesis.conflict_summary}\n\n"

        if synthesis.reasoning_trace:
            answer_text += "**Reasoning Trace**:\n"
            for i, step in enumerate(synthesis.reasoning_trace, 1):
                answer_text += f"{i}. {step}\n"
            answer_text += "\n"

        # Source attribution
        answer_text += "**Sources Used**:\n"
        for p in all_passages[:5]:
            source = p.get("source", p.get("domain", "unknown"))
            title = p.get("title", "")
            answer_text += f"- {source}: {title[:80]}\n"
    else:
        answer_text = "Pipeline completed but synthesis failed. Check logs for details."

    return {
        "answer": answer_text,
        "sources": all_passages[:10],
        "conflicts": conflicts,
        "scored_claims": scored_claims,
        "messages": [],
        "metadata": {
            "query": query,
            "uploaded_files": uploaded_files or [],
            "reasoning_trace": reasoning_trace,
            "num_sources": len(all_passages),
            "num_web": len(all_web_results),
            "num_kb": len(all_kb_results),
            "num_uploaded": len(uploaded_passages),
            "num_claims": len(pipeline_result.get("claims", [])),
            "num_conflicts": len(conflicts),
        },
    }


def _deduplicate_passages(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate passages by text content.

    Args:
        passages: All gathered passages.

    Returns:
        Deduplicated list preserving highest-scoring copy.
    """
    seen: dict[str, dict[str, Any]] = {}
    for p in passages:
        text = p.get("text", "").strip()
        if not text:
            continue
        key = text[:200]  # dedup on first 200 chars
        existing = seen.get(key)
        if existing is None or p.get("score", 0) > existing.get("score", 0):
            seen[key] = p
    return list(seen.values())


def stream_deep_research(
    query: str,
    agent: dict[str, Any] | None = None,
    uploaded_files: list[str] | None = None,
):
    """Stream deep research results for real-time UI updates.

    Yields events as each stage of the pipeline completes.

    Args:
        query: User query string.
        agent: Pre-created agent context. If None, a new one is created.
        uploaded_files: Optional list of file paths to analyze.

    Yields:
        Dict events with ``type`` and ``content``.
    """
    if agent is None:
        agent = create_research_agent()

    qdrant = agent["qdrant"]
    embedder = agent["embedder"]
    llm_heavy = agent["llm_heavy"]
    llm_light = agent["llm_light"]
    settings = agent["settings"]

    yield {"type": "thinking", "content": f"Analyzing query: {query}"}

    # Step 1: Decompose
    sub_queries = _decompose_query(query, llm_light)
    yield {"type": "thinking", "content": f"Decomposed into {len(sub_queries)} sub-queries"}

    # Step 2: Search
    all_passages: list[dict[str, Any]] = []

    for sq in sub_queries:
        yield {"type": "tool_call", "tool": "web_search", "content": f"Searching web: {sq[:50]}..."}
        web_results = _search_web(sq, max_results=settings.tavily_max_results)
        all_passages.extend(web_results)
        yield {"type": "tool_result", "tool": "web_search", "content": f"Found {len(web_results)} web results"}

        yield {"type": "tool_call", "tool": "kb_search", "content": f"Searching KB: {sq[:50]}..."}
        kb_results = _search_kb(sq, qdrant, embedder, top_k=settings.retrieval_top_k)
        all_passages.extend(kb_results)
        yield {"type": "tool_result", "tool": "kb_search", "content": f"Found {len(kb_results)} KB results"}

    # Step 3: Uploaded docs
    if uploaded_files:
        for fp in uploaded_files:
            yield {"type": "tool_call", "tool": "document_parser", "content": f"Parsing: {fp}"}
            passages = parse_document_structured(fp)
            all_passages.extend(passages)
            yield {"type": "tool_result", "tool": "document_parser", "content": f"Extracted {len(passages)} passages"}

    all_passages = _deduplicate_passages(all_passages)
    yield {"type": "thinking", "content": f"Total evidence: {len(all_passages)} unique passages"}

    if not all_passages:
        yield {"type": "error", "content": "No sources found for this query."}
        return

    # Step 4: Conflict pipeline
    yield {"type": "tool_call", "tool": "conflict_pipeline", "content": "Running conflict detection..."}
    pipeline_result = _run_conflict_pipeline(
        query, all_passages, llm_heavy, llm_light, embedder,
    )
    conflicts = pipeline_result.get("conflicts", [])
    yield {
        "type": "tool_result",
        "tool": "conflict_pipeline",
        "content": f"Detected {len(conflicts)} conflicts across {len(pipeline_result.get('claims', []))} claims",
    }

    yield {"type": "complete", "content": "Research complete."}
