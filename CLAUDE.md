# Deliberative RAG — Conflict-Aware Retrieval Reasoning Framework

A RAG system that detects contradictions between retrieved sources, builds a claim-level conflict graph, scores claims by temporal recency and source authority, and generates answers with explicit reasoning traces and calibrated confidence scores.

## Architecture

6-stage pipeline:

1. **Query Analysis** — decompose query into sub-queries
2. **Multi-Strategy Retrieval** — vector search + metadata filtering from Qdrant
3. **Claim Extraction** — break passages into atomic claims using LLM
4. **Conflict Graph Construction** — build typed graph (supports/contradicts/supersedes)
5. **Temporal-Authority Scoring** — score claims by recency + source credibility + graph evidence
6. **Deliberation Synthesis** — generate answer with reasoning trace and confidence score

## Tech Stack

- **Python 3.11+**
- **LangGraph** — agent orchestration
- **Qdrant** — vector storage (local mode)
- **sentence-transformers** — embeddings
- **NetworkX** — conflict graph
- **FastAPI** — API layer
- **Streamlit** — demo UI
- **Claude Sonnet via langchain-anthropic** — LLM calls

## Code Conventions

- All data models use **Pydantic v2**
- All LLM prompts stored as `.txt` files in `config/prompts/`
- All config values in `config/settings.py` — no magic numbers in code
- **Type hints everywhere**
- **Docstrings on all public functions**
- Tests in `tests/` using **pytest**

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
