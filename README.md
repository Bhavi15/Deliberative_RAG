<div align="center">

<br />

```
██████╗ ███████╗██╗     ██╗██████╗ ███████╗██████╗  █████╗ ████████╗███████╗
██╔══██╗██╔════╝██║     ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
██║  ██║█████╗  ██║     ██║██████╔╝█████╗  ██████╔╝███████║   ██║   █████╗
██║  ██║██╔══╝  ██║     ██║██╔══██╗██╔══╝  ██╔══██╗██╔══██║   ██║   ██╔══╝
██████╔╝███████╗███████╗██║██████╔╝███████╗██║  ██║██║  ██║   ██║   ███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
                                                                          RAG
```

### **Conflict-Aware Deep Research Agent**

*Standard RAG ignores contradictions. This one reasons about them.*

<br />

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-6--Stage_DAG-1C3C3C?style=flat-square)
![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-EF476F?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-Gemma_4_via_Ollama-000000?style=flat-square)
![MCP](https://img.shields.io/badge/Protocol-MCP_Tools-6C63FF?style=flat-square)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

</div>

---

## The Problem with Standard RAG

Naive RAG concatenates retrieved passages and hands them to an LLM — regardless of whether those passages **contradict each other**. The model blends them silently, producing overconfident answers with no mechanism to detect, flag, or reason about conflict.

**Deliberative RAG treats contradiction as a first-class signal**, not noise to suppress.

---

## How It Works

```
Query → Sub-query Decomposition → Hybrid Retrieval (Vector + BM25 + Web)
      → Atomic Claim Extraction → Conflict Graph Construction
      → Temporal-Authority Scoring → Calibrated Answer + Reasoning Trace
```

---

## Core AI Concepts

<br />

### 🧩 Atomic Claim Extraction
Retrieved passages are decomposed into **typed, self-contained assertions** rather than fed as raw text. Each claim carries a `ClaimType` (`fact | opinion | forecast | data_point`), a temporal marker, and a source reference — giving downstream stages something structured to reason about.

---

### 🕸️ Typed Conflict Graph — `NetworkX DiGraph`
Claims are embedded, clustered by cosine similarity, and pairwise-classified by an LLM into four relation types:

| Relation | Meaning |
|:---|:---|
| `SUPPORTS` | Corroborating evidence |
| `CONTRADICTS` | Mutually exclusive assertions |
| `SUPERSEDES` | Temporal update — newer claim overrides older |
| `UNRELATED` | No meaningful connection |

The graph encodes the **relational structure of evidence** before any synthesis begins. This is the primitive standard RAG doesn't have.

---

### ⚖️ Temporal-Authority Composite Scoring

Every claim receives a trust score from three orthogonal signals:

```
final_score = 0.35 × Temporal + 0.40 × Authority + 0.25 × Graph

Temporal  = exp(−0.693 × days_old / half_life)      # exponential decay
Authority = lookup(source_type)                      # SEC → 1.0, web → 0.55
Graph     = sigmoid(Σ SUPPORTS − Σ CONTRADICTS)     # community endorsement
```

> Any claim tagged `SUPERSEDES` forces the superseded claim's score to **0.1** — regardless of how many other claims support it.

---

### 🔀 Hybrid Retrieval with Reciprocal Rank Fusion

Three retrieval strategies run **in parallel** and are merged via RRF:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))   where k = 60
```

- **Qdrant** — dense vector search (`all-MiniLM-L6-v2`, 384-dim)
- **BM25** — lexical match for exact terms, proper nouns, domain jargon
- **Tavily** — real-time web retrieval for time-sensitive queries

For user-uploaded documents, BM25 candidate selection feeds a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) — no pre-computed embeddings needed.

---

### 🤖 Deep Research Agent — `deepagents`

Wraps the deterministic pipeline with a planning-capable agent:

- **Query decomposition** — breaks multi-hop questions into retrieval sub-tasks
- **Parallel subagent spawning** — independent research tracks run concurrently
- **Persistent cross-session memory** — no re-retrieval of known evidence
- **Deterministic tool fallback** — detects LLM tool-call failures; falls back to template-driven execution to prevent cascading errors

---

### 📊 Confidence Calibration — ECE

The system outputs a numerical `confidence_score ∈ [0, 1]` derived from the scoring function — **not generated post-hoc**. Calibration is measured via Expected Calibration Error:

```
ECE = Σ_b (|B_b| / n) × |accuracy(B_b) − confidence(B_b)|
```

A score of `0.8` should mean the answer is correct 80% of the time on those examples.

---

## Output Schema

Every response includes more than just an answer:

```python
class DeliberationResult(BaseModel):
    answer: str
    confidence: ConfidenceLevel          # HIGH | MODERATE | LOW
    confidence_score: float              # [0, 1] — calibrated
    reasoning_trace: list[str]           # explicit decision log per stage
    source_attribution: list[SourceAttribution]
    conflict_summary: str                # human-readable resolution narrative
```

---

## Evaluation

Benchmarked against three datasets with an ablation study isolating each scoring dimension:

| Dataset | Tests |
|:---|:---|
| `osunlp/ConflictQA` | Contradiction detection recall |
| `google/frames-benchmark` | Multi-hop reasoning across sources |
| `PatronusAI/financebench` | Numerical accuracy, financial domain |

| Ablation Variant | What It Isolates |
|:---|:---|
| No Temporal | Value of recency decay |
| No Authority | Value of source credibility |
| No Graph | Value of conflict detection |
| No Claims | Value of atomic decomposition |
| Baseline RAG | No conflict detection at all |

---

## Stack

```python
"langgraph"             # 6-stage stateful pipeline DAG
"deepagents"            # Planning + parallel subagent orchestration
"langchain-ollama"      # Local Gemma 4 inference (e4b heavy / e2b light)
"qdrant-client"         # Local vector store — no external service
"sentence-transformers" # all-MiniLM-L6-v2 embeddings + cross-encoder reranking
"rank-bm25"             # Lexical retrieval
"networkx"              # Typed conflict graph (DiGraph)
"pydantic"              # v2 — runtime type enforcement throughout
"fastapi"               # Async REST API
"mcp"                   # Composable tool protocol
"tavily-python"         # Web search
"streamlit"             # Demo UI + interactive conflict graph viz
"pyvis"                 # Conflict graph visualization
"pytest"                # 9 test modules, async support
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Pull models
ollama pull gemma4:e4b && ollama pull gemma4:e2b

# Demo UI
streamlit run demo/app.py

# REST API
uvicorn api.main:app --reload

# MCP server (Claude Desktop, Cursor, VS Code Copilot)
python -m src.mcp.server

# Evaluation + ablations
python -m evaluation.run_eval --sample 10
python -m evaluation.ablations
```

```bash
# .env
TAVILY_API_KEY=...
QDRANT_PATH=./qdrant_data
OLLAMA_BASE_URL=http://localhost:11434
```

---

<div align="center">

*Conflict isn't noise. It's the most important signal in the evidence.*

<br />

![Conflict-Aware](https://img.shields.io/badge/Conflict_Detection-First_Class_Signal-EF476F?style=flat-square)
![Calibrated](https://img.shields.io/badge/Answers-Calibrated_%2B_Traced-6C63FF?style=flat-square)

</div>
