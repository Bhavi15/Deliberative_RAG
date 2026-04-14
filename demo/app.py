"""Streamlit demo application for Deliberative RAG.

Features:
- **Deep Research Mode**: Uses the deepagents-powered research agent
  with iterative web search + KB search + conflict detection
- **Classic Pipeline Mode**: Original 6-stage LangGraph pipeline
- **Document Upload**: Analyze uploaded PDFs/docs for conflicts
- **Streaming**: Real-time reasoning trace display

Run with::

    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

# Suppress noisy torchvision/transformers warnings from Streamlit file watcher
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*__path__.*")

import streamlit as st

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so settings are available
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from src.schemas import (  # noqa: E402
    Claim,
    ClaimType,
    ConfidenceLevel,
    ConflictEdge,
    DeliberationResult,
    RelationType,
    ScoredClaim,
    SourceAttribution,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Deliberative RAG",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Pre-loaded example queries
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES: dict[str, list[dict[str, str]]] = {
    "ConflictQA": [
        {
            "query": "What is Kathy Saltzman's occupation?",
            "description": "Parametric memory says musician; counter-evidence says politician",
        },
        {
            "query": "What is Eleanor Davis's occupation?",
            "description": "Conflicting claims about career (artist vs other)",
        },
        {
            "query": "What is Javier Alva Orlandini's occupation?",
            "description": "Contradictory evidence about professional background",
        },
    ],
    "FRAMES": [
        {
            "query": "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name? ",
            "description": "Multi-hop reasoning across historical facts",
        },
        {
            "query": "How many years earlier would Punxsutawney Phil have to be canonically alive to have made a Groundhog Day prediction in the same state as the US capitol?",
            "description": "Multi-step numerical reasoning",
        },
    ],
    "FinanceBench": [
        {
            "query": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
            "description": "Precise financial data extraction from cash flow",
        },
        {
            "query": "Is 3M a capital-intensive business based on FY2022 data?",
            "description": "Analytical financial reasoning",
        },
    ],
    "Deep Research (Web)": [
        {
            "query": "What are the current contradictions in scientific consensus about the health effects of intermittent fasting?",
            "description": "Web search + conflict detection on live sources",
        },
        {
            "query": "Do different sources agree on the current inflation rate in the US?",
            "description": "Real-time data conflict detection",
        },
    ],
}

# ---------------------------------------------------------------------------
# Demo result generation
# ---------------------------------------------------------------------------


def _load_dataset_examples() -> dict[str, dict]:
    """Load the master eval dataset indexed by question text."""
    dataset_path = PROJECT_ROOT / "data" / "evaluation" / "master_eval_dataset.json"
    if not dataset_path.exists():
        return {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["question"]: d for d in data}


@st.cache_data
def get_dataset_index() -> dict[str, dict]:
    """Cached dataset index."""
    return _load_dataset_examples()


def run_deep_research_pipeline(
    query: str,
    uploaded_files: list[str] | None = None,
) -> dict[str, Any]:
    """Run the deep research agent (deepagents + web search + KB + conflict detection).

    Args:
        query: User query string.
        uploaded_files: Optional list of uploaded file paths.

    Returns:
        Dict with result data for rendering.
    """
    try:
        from src.agent.deep_research import run_deep_research
        result = run_deep_research(query, uploaded_files=uploaded_files)

        return {
            "answer": result.get("answer", "No result"),
            "messages": result.get("messages", []),
            "metadata": result.get("metadata", {}),
            "is_live": True,
            "mode": "deep_research",
        }
    except Exception as exc:
        return {
            "answer": f"Deep research failed: {exc}",
            "messages": [],
            "metadata": {"error": str(exc)},
            "is_live": False,
            "mode": "deep_research",
        }


def run_classic_pipeline(
    query: str,
    half_life: int,
    authority_weights: dict[str, bool],
) -> dict[str, Any]:
    """Run the classic 6-stage deliberative pipeline.

    Args:
        query: User query string.
        half_life: Temporal decay half-life in days.
        authority_weights: Which authority types to include.

    Returns:
        Dict with result data for rendering.
    """
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    if ollama_url:
        return _run_live_pipeline(query, half_life, authority_weights)

    return _build_demo_result(query)


def _run_live_pipeline(
    query: str,
    half_life: int,
    authority_weights: dict[str, bool],
) -> dict[str, Any]:
    """Run the real pipeline with live LLM calls."""
    from src.agent.graph import run_query_full
    from src.utils.llm import get_llm
    from src.vectorstore.embeddings import EmbeddingModel
    from src.vectorstore.qdrant_client import QdrantManager

    llm_heavy = get_llm("heavy")
    llm_light = get_llm("light")
    embedder = EmbeddingModel()
    qdrant = QdrantManager()

    try:
        full_state = run_query_full(
            query, llm_heavy=llm_heavy, qdrant=qdrant,
            embedder=embedder, llm_light=llm_light, top_k=3,
        )
        result: DeliberationResult = full_state["final_answer"]
        scored_claims: list[ScoredClaim] = full_state.get("scored_claims", [])
        raw_edges = full_state.get("conflict_graph", {}).get("edges", [])
        conflict_edges = [ConflictEdge(**e) for e in raw_edges]
        documents = full_state.get("retrieved_documents", [])
        return {
            "result": result,
            "scored_claims": scored_claims,
            "conflict_edges": conflict_edges,
            "documents": documents,
            "is_live": True,
            "mode": "classic",
        }
    finally:
        qdrant.close()


def _build_demo_result(query: str) -> dict[str, Any]:
    """Build a synthetic result from the evaluation dataset for demo purposes."""
    dataset = get_dataset_index()
    example = dataset.get(query)

    if example is None:
        return _build_fallback_result(query)

    passages = example.get("evidence_passages", [])
    has_conflict = example.get("has_known_conflict", False)
    gt = example.get("ground_truth_answer", "Unknown")
    if isinstance(gt, list):
        gt_text = gt[0] if gt else "Unknown"
    else:
        gt_text = gt

    scored_claims: list[ScoredClaim] = []
    for i, p in enumerate(passages):
        text = p.get("text", "")[:200]
        source = p.get("source", "unknown")
        is_correct = not (has_conflict and source == "parametric_memory")

        scored_claims.append(ScoredClaim(
            claim_id=f"claim_{i:03d}",
            text=text,
            claim_type=ClaimType.FACT,
            source_doc_id=f"doc_{source}_{i}",
            temporal_marker=None,
            confidence_in_extraction=0.90 if is_correct else 0.85,
            temporal_score=0.9 if is_correct else 0.7,
            authority_score=0.8 if source != "parametric_memory" else 0.4,
            graph_evidence_score=0.75 if is_correct else 0.3,
            final_score=0.82 if is_correct else 0.45,
        ))

    conflict_edges: list[ConflictEdge] = []
    if has_conflict and len(scored_claims) >= 2:
        conflict_edges.append(ConflictEdge(
            source_claim_id=scored_claims[0].claim_id,
            target_claim_id=scored_claims[1].claim_id,
            relation=RelationType.CONTRADICTS,
            confidence=0.92,
            reasoning="Sources provide mutually exclusive claims about the subject.",
        ))

    for i in range(len(scored_claims)):
        for j in range(i + 1, len(scored_claims)):
            if scored_claims[i].final_score > 0.6 and scored_claims[j].final_score > 0.6:
                if not any(
                    e.source_claim_id in (scored_claims[i].claim_id, scored_claims[j].claim_id)
                    and e.target_claim_id in (scored_claims[i].claim_id, scored_claims[j].claim_id)
                    for e in conflict_edges
                ):
                    conflict_edges.append(ConflictEdge(
                        source_claim_id=scored_claims[i].claim_id,
                        target_claim_id=scored_claims[j].claim_id,
                        relation=RelationType.SUPPORTS,
                        confidence=0.80,
                        reasoning="Claims are consistent and mutually reinforcing.",
                    ))

    if has_conflict:
        conf = ConfidenceLevel.HIGH
        conf_score = 0.88
        trace = [
            f"Decomposed query into {max(1, len(passages))} sub-queries.",
            f"Retrieved {len(passages)} source documents.",
            f"Extracted {len(scored_claims)} atomic claims from passages.",
            "Detected contradiction between sources (parametric vs counter-evidence).",
            f"Resolved conflict in favour of higher-scored claim: '{gt_text}'.",
            f"Final answer confidence: {conf_score:.0%} (conflict resolved with clear evidence).",
        ]
        conflict_summary = (
            "Sources disagreed on the key fact. Parametric memory claim "
            "was contradicted by counter-evidence with higher authority "
            "and graph support. Resolved in favour of the counter-evidence."
        )
    else:
        conf = ConfidenceLevel.HIGH
        conf_score = 0.91
        trace = [
            f"Decomposed query into {max(1, len(passages))} sub-queries.",
            f"Retrieved {len(passages)} source documents.",
            f"Extracted {len(scored_claims)} claims — all consistent.",
            "No contradictions detected in the conflict graph.",
            f"Synthesized answer from top-scored claims with confidence {conf_score:.0%}.",
        ]
        conflict_summary = ""

    attributions = [
        SourceAttribution(
            claim_id=sc.claim_id,
            source_doc_id=sc.source_doc_id,
            relevance=round(sc.final_score, 2),
        )
        for sc in sorted(scored_claims, key=lambda s: s.final_score, reverse=True)[:3]
    ]

    result = DeliberationResult(
        query=query,
        answer=gt_text,
        confidence=conf,
        confidence_score=conf_score,
        reasoning_trace=trace,
        source_attribution=attributions,
        conflict_summary=conflict_summary,
    )

    documents = [
        {
            "id": f"doc_{p.get('source', 'unknown')}_{i}",
            "text": p.get("text", ""),
            "source": p.get("source", "unknown"),
            "score": round(0.9 - i * 0.05, 2),
        }
        for i, p in enumerate(passages)
    ]

    return {
        "result": result,
        "scored_claims": scored_claims,
        "conflict_edges": conflict_edges,
        "documents": documents,
        "is_live": False,
        "mode": "classic",
    }


def _build_fallback_result(query: str) -> dict[str, Any]:
    """Fallback when query is not in the dataset and no API key is set."""
    result = DeliberationResult(
        query=query,
        answer="This query is not in the pre-loaded examples. Make sure Ollama is running for live queries, or try one of the example queries.",
        confidence=ConfidenceLevel.LOW,
        confidence_score=0.1,
        reasoning_trace=["Query not found in pre-loaded dataset."],
        source_attribution=[],
        conflict_summary="",
    )
    return {
        "result": result,
        "scored_claims": [],
        "conflict_edges": [],
        "documents": [],
        "is_live": False,
        "mode": "classic",
    }


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Render the page title and introductory description."""
    st.title("Deliberative RAG")
    st.caption(
        "Conflict-Aware Retrieval Reasoning — detects contradictions between "
        "sources, scores claims by recency and authority, and generates answers "
        "with explicit reasoning traces. Powered by Deep Research Agent, "
        "Web Search, and MCP protocol."
    )


def render_sidebar_settings() -> tuple[str, int, dict[str, bool], list[str]]:
    """Render the left panel settings and return current values.

    Returns:
        Tuple of (mode, half_life, authority_weights, uploaded_file_paths).
    """
    st.sidebar.header("Research Mode")

    mode = st.sidebar.radio(
        "Select mode",
        options=["Deep Research", "Classic Pipeline"],
        index=0,
        help=(
            "**Deep Research**: Uses deepagents with iterative web + KB search "
            "and conflict detection.\n\n"
            "**Classic Pipeline**: Original 6-stage LangGraph pipeline on "
            "indexed knowledge base only."
        ),
    )

    st.sidebar.markdown("---")

    # Document upload
    st.sidebar.header("Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents for conflict analysis",
        type=["pdf", "txt", "md", "html"],
        accept_multiple_files=True,
        help="Upload PDFs or text files. Claims will be extracted and "
             "cross-referenced against web + KB sources.",
    )

    # Save uploaded files to temp directory
    uploaded_paths: list[str] = []
    if uploaded_files:
        for uf in uploaded_files:
            temp_path = Path(tempfile.gettempdir()) / uf.name
            temp_path.write_bytes(uf.getbuffer())
            uploaded_paths.append(str(temp_path))

    st.sidebar.markdown("---")
    st.sidebar.header("Settings")

    half_life = st.sidebar.slider(
        "Temporal half-life (days)",
        min_value=7,
        max_value=365,
        value=90,
        step=7,
        help="How quickly source recency decays. Lower = more weight on recent sources.",
    )

    st.sidebar.subheader("Authority weights")
    weights = {
        "official": st.sidebar.checkbox("Official sources", value=True),
        "peer_reviewed": st.sidebar.checkbox("Peer-reviewed", value=True),
        "encyclopedic": st.sidebar.checkbox("Encyclopedic", value=True),
        "news": st.sidebar.checkbox("News sources", value=True),
        "blog": st.sidebar.checkbox("Blog posts", value=False),
    }

    return mode, half_life, weights, uploaded_paths


def render_example_queries() -> str | None:
    """Render pre-loaded example query buttons in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("Example Queries")

    for dataset_name, examples in EXAMPLE_QUERIES.items():
        st.sidebar.subheader(dataset_name)
        for ex in examples:
            if st.sidebar.button(
                ex["query"][:55] + ("..." if len(ex["query"]) > 55 else ""),
                key=f"ex_{hash(ex['query'])}",
                help=ex["description"],
                use_container_width=True,
            ):
                return ex["query"]

    return None


def render_source_documents(documents: list[dict]) -> None:
    """Render source documents with relevance scores in the sidebar."""
    if not documents:
        return

    st.sidebar.markdown("---")
    st.sidebar.header("Source Documents")

    for i, doc in enumerate(documents):
        score = doc.get("score", 0.0)
        source = doc.get("source", "unknown")
        text = doc.get("text", "")

        if score >= 0.8:
            badge = f":green[{score:.2f}]"
        elif score >= 0.5:
            badge = f":orange[{score:.2f}]"
        else:
            badge = f":red[{score:.2f}]"

        with st.sidebar.expander(f"Doc {i + 1} ({source}) -- {badge}"):
            st.write(text[:500])
            if len(text) > 500:
                st.caption(f"... ({len(text)} chars total)")


def render_confidence_badge(result: DeliberationResult) -> None:
    """Render the answer with a colored confidence indicator."""
    conf = result.confidence.value
    score = result.confidence_score

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Answer")
        st.markdown(f"**{result.answer}**")
    with col2:
        st.metric(
            label="Confidence",
            value=f"{score:.0%}",
            delta=conf.upper(),
        )

    if result.conflict_summary:
        st.warning(f"**Conflicts detected:** {result.conflict_summary}")


def render_deep_research_result(output: dict[str, Any]) -> None:
    """Render deep research agent output with streaming-style display."""
    answer = output.get("answer", "")
    metadata = output.get("metadata", {})

    st.subheader("Deep Research Result")

    if metadata.get("error"):
        st.error(f"Research error: {metadata['error']}")
        return

    # Display the answer
    st.markdown(answer)

    # Metadata
    with st.expander("Research Metadata"):
        st.json(metadata)


def render_conflict_graph(
    scored_claims: list[ScoredClaim],
    conflict_edges: list[ConflictEdge],
) -> None:
    """Render an interactive conflict graph using pyvis."""
    if not scored_claims:
        st.info("No claims to display in the conflict graph.")
        return

    from pyvis.network import Network

    net = Network(
        height="420px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    for sc in scored_claims:
        score = sc.final_score
        if score >= 0.7:
            color = "#4CAF50"
        elif score >= 0.4:
            color = "#FF9800"
        else:
            color = "#F44336"

        label = sc.text[:60] + ("..." if len(sc.text) > 60 else "")
        title = (
            f"<b>{sc.claim_id}</b><br>"
            f"Text: {sc.text}<br>"
            f"Source: {sc.source_doc_id}<br>"
            f"Score: {sc.final_score:.2f}<br>"
            f"  temporal={sc.temporal_score:.2f} "
            f"authority={sc.authority_score:.2f} "
            f"graph={sc.graph_evidence_score:.2f}"
        )
        net.add_node(
            sc.claim_id,
            label=label,
            title=title,
            color=color,
            size=20 + score * 20,
            font={"size": 11, "color": "white"},
        )

    edge_colors = {
        RelationType.SUPPORTS: "#4CAF50",
        RelationType.CONTRADICTS: "#F44336",
        RelationType.SUPERSEDES: "#FF9800",
    }

    for edge in conflict_edges:
        color = edge_colors.get(edge.relation, "#9E9E9E")
        width = 2 if edge.relation == RelationType.SUPPORTS else 3
        dashes = edge.relation == RelationType.SUPERSEDES

        if edge.source_claim_id not in [n["id"] for n in net.nodes]:
            net.add_node(edge.source_claim_id, label=edge.source_claim_id, color="#666")
        if edge.target_claim_id not in [n["id"] for n in net.nodes]:
            net.add_node(edge.target_claim_id, label=edge.target_claim_id, color="#666")

        net.add_edge(
            edge.source_claim_id,
            edge.target_claim_id,
            label=edge.relation.value,
            color=color,
            width=width,
            dashes=dashes,
            title=edge.reasoning,
            arrows="to",
            font={"size": 10, "color": color, "strokeWidth": 0},
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8",
    ) as f:
        net.write_html(f.name)
        html_path = f.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    import streamlit.components.v1 as components
    components.html(html_content, height=450, scrolling=False)

    legend_cols = st.columns(5)
    with legend_cols[0]:
        st.markdown(":green[**Green node**] = high score")
    with legend_cols[1]:
        st.markdown(":orange[**Orange node**] = mid score")
    with legend_cols[2]:
        st.markdown(":red[**Red node**] = low score")
    with legend_cols[3]:
        st.markdown(":red[**Red edge**] = contradicts")
    with legend_cols[4]:
        st.markdown(":green[**Green edge**] = supports")


def render_reasoning_trace(result: DeliberationResult) -> None:
    """Render the reasoning trace in an expandable section."""
    if not result.reasoning_trace:
        return

    with st.expander("Reasoning Trace", expanded=True):
        for i, step in enumerate(result.reasoning_trace, 1):
            st.markdown(f"**Step {i}.** {step}")


def render_scored_claims(scored_claims: list[ScoredClaim]) -> None:
    """Render scored claims in an expandable table."""
    if not scored_claims:
        return

    with st.expander(f"Scored Claims ({len(scored_claims)})", expanded=False):
        for sc in sorted(scored_claims, key=lambda s: s.final_score, reverse=True):
            score_bar = "\u2588" * int(sc.final_score * 10)
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{sc.claim_id}** -- {sc.text[:120]}")
                st.caption(
                    f"Source: `{sc.source_doc_id}` | "
                    f"temporal={sc.temporal_score:.2f} "
                    f"authority={sc.authority_score:.2f} "
                    f"graph={sc.graph_evidence_score:.2f}"
                )
            with col2:
                st.markdown(f"**{sc.final_score:.2f}**")
                st.caption(score_bar)


def render_source_attribution(result: DeliberationResult) -> None:
    """Render source attribution."""
    if not result.source_attribution:
        return

    with st.expander("Source Attribution", expanded=False):
        for sa in result.source_attribution:
            st.markdown(
                f"- **{sa.claim_id}** from `{sa.source_doc_id}` "
                f"-- relevance: {sa.relevance:.2f}"
            )


def render_architecture_info() -> None:
    """Render architecture information in an expandable section."""
    with st.expander("System Architecture", expanded=False):
        st.markdown("""
**Deep Research Mode** uses a multi-layer architecture:

1. **Deep Agent** (LangChain `deepagents`) — Plans research, spawns subagents
2. **Hybrid Retrieval** — BM25 + Vector (Qdrant) + Web (Tavily) with RRF fusion
3. **Claim Extraction** — LLM breaks passages into atomic claims
4. **Conflict Graph** — NetworkX graph with SUPPORTS/CONTRADICTS/SUPERSEDES edges
5. **Temporal-Authority Scoring** — Exponential recency decay + domain authority weights
6. **Deliberative Synthesis** — Generates answer with reasoning trace + confidence

**Protocols:**
- **MCP** (Model Context Protocol) — Exposes pipeline as composable tools for Claude Desktop, Cursor, etc.
        """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    render_header()

    # Sidebar
    mode, half_life, authority_weights, uploaded_paths = render_sidebar_settings()
    selected_example = render_example_queries()

    # Main query input
    st.markdown("---")

    if selected_example and "current_query" not in st.session_state:
        st.session_state["current_query"] = selected_example
    if selected_example:
        st.session_state["current_query"] = selected_example

    query = st.text_input(
        "Enter your query",
        value=st.session_state.get("current_query", ""),
        placeholder="e.g., What is Kathy Saltzman's occupation?",
    )

    # Mode indicator
    if mode == "Deep Research":
        st.info(
            "**Deep Research Mode** — Agent will search web + knowledge base, "
            "iteratively gather sources, and run conflict detection."
        )
    else:
        st.info(
            "**Classic Pipeline** — 6-stage LangGraph pipeline on indexed "
            "knowledge base."
        )

    # Upload indicator
    if uploaded_paths:
        st.success(f"Uploaded {len(uploaded_paths)} document(s) for analysis.")

    run_button = st.button("Run Research", type="primary", use_container_width=True)

    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if tavily_key and not tavily_key.startswith("tvly-xxx"):
            st.caption("Web Search: Active")
        else:
            st.caption("Web Search: Not configured (set TAVILY_API_KEY)")
    with col2:
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        st.caption(f"LLM: Ollama at {ollama_url}")
    with col3:
        st.caption(f"Mode: {mode}")

    st.markdown("---")

    # Architecture info
    render_architecture_info()

    # Run pipeline on submit
    if run_button and query:
        if mode == "Deep Research":
            with st.spinner("Running deep research agent..."):
                pipeline_output = run_deep_research_pipeline(
                    query, uploaded_files=uploaded_paths or None,
                )

            render_deep_research_result(pipeline_output)

        else:
            # Classic pipeline mode
            with st.spinner("Running deliberative pipeline..."):
                pipeline_output = run_classic_pipeline(
                    query, half_life, authority_weights,
                )

            result: DeliberationResult = pipeline_output["result"]
            scored_claims: list[ScoredClaim] = pipeline_output["scored_claims"]
            conflict_edges: list[ConflictEdge] = pipeline_output["conflict_edges"]
            documents: list[dict] = pipeline_output["documents"]

            if not pipeline_output.get("is_live", False):
                st.info("Showing pre-computed demo results. Start Ollama for live pipeline.")

            render_confidence_badge(result)

            st.markdown("---")

            st.subheader("Conflict Graph")
            render_conflict_graph(scored_claims, conflict_edges)

            st.markdown("---")

            render_reasoning_trace(result)
            render_scored_claims(scored_claims)
            render_source_attribution(result)

            render_source_documents(documents)

    elif run_button and not query:
        st.warning("Please enter a query or select an example from the sidebar.")


if __name__ == "__main__":
    main()
