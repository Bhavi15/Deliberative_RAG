# Deliberative RAG — Conflict-Aware Deep Research Agent

An agentic RAG system that detects contradictions between retrieved sources, builds a claim-level conflict graph, scores claims by temporal recency and source authority, and generates answers with explicit reasoning traces and calibrated confidence scores.

## Architecture

Multi-layer architecture with deep research agent orchestration:

### Layer 1 — Deep Research Agent (deepagents)
- **Research Planning** — decomposes queries, decides retrieval strategy
- **Iterative Search** — multi-round web + KB search until sufficient evidence gathered
- **Subagent Spawning** — parallel research tracks for complex queries
- **Persistent Memory** — cross-session research context

### Layer 2 — Hybrid Retrieval
- **Vector Search** — Qdrant (cosine similarity, all-MiniLM-L6-v2 embeddings)
- **BM25 Lexical Search** — keyword matching on indexed corpus
- **Web Search** — Tavily API for real-time web results
- **Reciprocal Rank Fusion** — combines all ranked lists into unified results
- **Vectorless Retrieval** — BM25 + cross-encoder reranking for uploaded docs (no vector DB needed)

### Layer 3 — 6-Stage Deliberative Pipeline
1. **Query Analysis** — decompose query into sub-queries
2. **Multi-Strategy Retrieval** — hybrid search from Qdrant + web + BM25
3. **Claim Extraction** — break passages into atomic claims using LLM
4. **Conflict Graph Construction** — build typed graph (supports/contradicts/supersedes)
5. **Temporal-Authority Scoring** — score claims by recency + source credibility + graph evidence
6. **Deliberation Synthesis** — generate answer with reasoning trace and confidence score

### Layer 4 — Protocol Integration
- **MCP Server** — exposes pipeline as composable tools for any MCP-compatible AI app
- **FastAPI** — REST API with deep research and classic pipeline modes

## Tech Stack

- **Python 3.11+**
- **LangGraph** — agent orchestration (classic pipeline)
- **deepagents** — deep research agent with planning + subagents
- **Qdrant** — vector storage (local mode)
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **rank-bm25** — BM25 lexical search
- **Tavily** — web search API
- **NetworkX** — conflict graph
- **pymupdf** — PDF document parsing
- **MCP SDK** — Model Context Protocol server
- **FastAPI** — REST API layer
- **Streamlit** — demo UI
- **Ollama via langchain-ollama** — LLM calls (dual-tier: gemma4:e4b heavy + gemma4:e2b light)

## Code Conventions

- All data models use **Pydantic v2**
- All LLM prompts stored as `.txt` files in `config/prompts/`
- All config values in `config/settings.py` — no magic numbers in code
- **Type hints everywhere**
- **Docstrings on all public functions**
- Tests in `tests/` using **pytest**

## Key Modules

| Module | Purpose |
|---|---|
| `src/agent/deep_research.py` | Deep research agent (deepagents) |
| `src/agent/graph.py` | Classic LangGraph 6-stage pipeline |
| `src/pipeline/hybrid_retriever.py` | BM25 + Vector + Web search with RRF |
| `src/pipeline/vectorless_retriever.py` | Vectorless retrieval: BM25 + cross-encoder reranking |
| `src/tools/web_search.py` | Tavily web search tool |
| `src/tools/document_parser.py` | PDF/text document parsing |
| `src/mcp/server.py` | MCP server exposing pipeline tools |
| `api/routes.py` | FastAPI REST endpoints |
| `demo/app.py` | Streamlit demo UI |

## Datasets

Pre-built datasets from HuggingFace:

| Dataset | HuggingFace ID | Purpose |
|---|---|---|
| ConflictQA | `osunlp/ConflictQA` | Primary contradiction detection |
| FRAMES | `google/frames-benchmark` | Multi-hop reasoning evaluation |
| FinanceBench | `PatronusAI/financebench` | Financial domain testing |

## Evaluation Metrics

1. **Factual Accuracy** — does the answer match ground truth?
2. **Conflict Detection Recall** — did the system catch known contradictions?
3. **Confidence Calibration** — do stated confidence levels correlate with actual accuracy?

## Running

```bash
# Demo UI (Deep Research + Classic modes)
streamlit run demo/app.py

# API server
uvicorn api.main:app --reload

# MCP server (for Claude Desktop, Cursor, etc.)
python -m src.mcp.server

# Evaluation
python -m evaluation.run_eval --sample 10

# Index data
python -m src.vectorstore.indexer
```
