"""Download and unify evaluation datasets into a single master JSON file.

Downloads:
- ConflictQA  (osunlp/ConflictQA, config ConflictQA-popQA-chatgpt) — 100 samples
- FRAMES      (google/frames-benchmark)                             —  50 samples
- FinanceBench(PatronusAI/financebench)                             —  50 samples

Each record is converted to a unified schema and saved to
data/evaluation/master_eval_dataset.json.

Usage:
    python data/download_datasets.py
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download

OUTPUT_DIR = Path(__file__).parent / "evaluation"
OUTPUT_PATH = OUTPUT_DIR / "master_eval_dataset.json"


# ---------------------------------------------------------------------------
# Unified record schema
# ---------------------------------------------------------------------------

def _make_record(
    *,
    source_dataset: str,
    question: str,
    ground_truth_answer: str | list[str],
    evidence_passages: list[dict],
    has_known_conflict: bool,
    metadata: dict,
) -> dict:
    """Build one unified evaluation record.

    Args:
        source_dataset: Which benchmark this example comes from.
        question: The evaluation question text.
        ground_truth_answer: Gold answer — string or list of acceptable strings.
        evidence_passages: List of dicts with at minimum ``"text"`` and ``"source"`` keys.
        has_known_conflict: True when the example is designed to contain contradictions.
        metadata: Arbitrary extra fields from the original dataset.

    Returns:
        A dict matching the unified evaluation schema.
    """
    return {
        "id": f"{source_dataset}_{uuid.uuid4().hex[:8]}",
        "source_dataset": source_dataset,
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "evidence_passages": evidence_passages,
        "has_known_conflict": has_known_conflict,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# ConflictQA
# ---------------------------------------------------------------------------

def download_conflictqa(n: int = 100) -> list[dict]:
    """Download ConflictQA and convert the first *n* examples.

    Each example contains a question with two conflicting answers:
    ``memory_answer`` (from LLM parametric memory) and ``counter_answer``
    (the factually correct counter-evidence). We store both evidence
    passages so the pipeline can detect the contradiction.

    Args:
        n: Number of examples to take.

    Returns:
        List of unified evaluation records.
    """
    print(f"  Downloading ConflictQA (taking {n})...")
    path = hf_hub_download(
        "osunlp/ConflictQA",
        "conflictQA-popQA-chatgpt.json",
        repo_type="dataset",
    )

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            records.append(
                _make_record(
                    source_dataset="conflict_qa",
                    question=row["question"],
                    ground_truth_answer=row["ground_truth"],
                    evidence_passages=[
                        {
                            "text": row["parametric_memory"],
                            "source": "parametric_memory",
                            "aligned_evidence": row.get("parametric_memory_aligned_evidence", ""),
                        },
                        {
                            "text": row["counter_memory"],
                            "source": "counter_memory",
                            "aligned_evidence": row.get("counter_memory_aligned_evidence", ""),
                        },
                    ],
                    has_known_conflict=True,
                    metadata={
                        "memory_answer": row["memory_answer"],
                        "counter_answer": row["counter_answer"],
                        "popularity": row.get("popularity"),
                    },
                )
            )

    print(f"    -> {len(records)} records loaded")
    return records


# ---------------------------------------------------------------------------
# FRAMES
# ---------------------------------------------------------------------------

def download_frames(n: int = 50) -> list[dict]:
    """Download FRAMES benchmark and convert the first *n* examples.

    FRAMES questions are multi-hop, requiring reasoning across 2-15
    Wikipedia articles. We store the Wikipedia links as evidence passages.

    Args:
        n: Number of examples to take.

    Returns:
        List of unified evaluation records.
    """
    print(f"  Downloading FRAMES (taking {n})...")
    path = hf_hub_download(
        "google/frames-benchmark",
        "test.tsv",
        repo_type="dataset",
    )

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= n:
                break

            # Gather non-empty Wikipedia links into evidence passages
            wiki_links: list[dict] = []
            for col in row:
                if col.startswith("wikipedia_link_") and row[col]:
                    wiki_links.append({"text": "", "source": row[col]})

            records.append(
                _make_record(
                    source_dataset="frames",
                    question=row["Prompt"],
                    ground_truth_answer=row["Answer"],
                    evidence_passages=wiki_links,
                    has_known_conflict=False,
                    metadata={
                        "reasoning_types": row.get("reasoning_types", ""),
                    },
                )
            )

    print(f"    -> {len(records)} records loaded")
    return records


# ---------------------------------------------------------------------------
# FinanceBench
# ---------------------------------------------------------------------------

def download_financebench(n: int = 50) -> list[dict]:
    """Download FinanceBench and convert the first *n* examples.

    FinanceBench contains financial QA pairs with evidence text extracted
    from SEC filings.

    Args:
        n: Number of examples to take.

    Returns:
        List of unified evaluation records.
    """
    print(f"  Downloading FinanceBench (taking {n})...")
    path = hf_hub_download(
        "PatronusAI/financebench",
        "financebench_merged.jsonl",
        repo_type="dataset",
    )

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)

            evidence_list: list[dict] = []
            raw_evidence = row.get("evidence", [])
            if isinstance(raw_evidence, list):
                for ev in raw_evidence:
                    if isinstance(ev, dict):
                        evidence_list.append({
                            "text": ev.get("evidence_text", ""),
                            "source": row.get("doc_link", ""),
                        })

            records.append(
                _make_record(
                    source_dataset="finance_bench",
                    question=row["question"],
                    ground_truth_answer=row["answer"],
                    evidence_passages=evidence_list,
                    has_known_conflict=False,
                    metadata={
                        "company": row.get("company", ""),
                        "doc_name": row.get("doc_name", ""),
                        "question_type": row.get("question_type", ""),
                        "question_reasoning": row.get("question_reasoning", ""),
                        "justification": row.get("justification", ""),
                        "gics_sector": row.get("gics_sector", ""),
                        "doc_period": row.get("doc_period", ""),
                    },
                )
            )

    print(f"    -> {len(records)} records loaded")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download all datasets, merge them, and write to disk."""
    print("Downloading evaluation datasets...\n")

    conflict_qa = download_conflictqa(n=100)
    frames = download_frames(n=50)
    finance_bench = download_financebench(n=50)

    master = conflict_qa + frames + finance_bench

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDone. Saved {len(master)} records to {OUTPUT_PATH}")
    print(f"  ConflictQA:   {len(conflict_qa)}")
    print(f"  FRAMES:       {len(frames)}")
    print(f"  FinanceBench: {len(finance_bench)}")


if __name__ == "__main__":
    main()
