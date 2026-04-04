"""Evaluation runner script.

Evaluates both the deliberative RAG system and the baseline RAG pipeline
across the master evaluation dataset and writes results to
``evaluation/results/``.

Usage::

    python -m evaluation.run_eval                     # full run (200 examples)
    python -m evaluation.run_eval --sample 10         # quick smoke-test
    python -m evaluation.run_eval --dataset conflict_qa --sample 20
    python -m evaluation.run_eval --baseline-only --sample 5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from tqdm import tqdm

from evaluation.baseline_rag import BaselineRAG
from evaluation.metrics import (
    calibration_error,
    confidence_calibration,
    conflict_detection_recall,
    summary_report,
)
from src.agent.graph import run_query_full
from src.schemas import ConflictEdge, DeliberationResult
from src.utils.llm import LLMClient
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()

DATASET_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "master_eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run evaluation comparing Deliberative RAG vs. Baseline.",
    )
    parser.add_argument(
        "--dataset",
        choices=["conflict_qa", "frames", "finance_bench", "all"],
        default="all",
        help="Which benchmark subset to evaluate on (default: all).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Cap on number of examples (for quick testing).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip the deliberative system; only run the baseline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve per query (default: 5).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_name: str = "all",
    max_samples: int | None = None,
) -> list[dict]:
    """Load and optionally filter the master evaluation dataset.

    Args:
        dataset_name: Filter to a specific subset, or ``"all"``.
        max_samples: Cap on number of examples returned.

    Returns:
        List of evaluation example dicts.
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    if dataset_name != "all":
        data = [d for d in data if d["source_dataset"] == dataset_name]

    if max_samples is not None:
        data = data[:max_samples]

    log.info(
        "dataset_loaded",
        total=len(data),
        filter=dataset_name,
        sample=max_samples,
    )
    return data


# ---------------------------------------------------------------------------
# Per-example runners
# ---------------------------------------------------------------------------


def run_deliberative_example(
    example: dict,
    llm: LLMClient,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run one example through the deliberative RAG pipeline.

    Returns a dict with ``result`` (DeliberationResult), ``conflict_edges``
    (list of ConflictEdge), and ``error`` (str or None).
    """
    try:
        full_state = run_query_full(
            example["question"],
            llm=llm,
            qdrant=qdrant,
            embedder=embedder,
            top_k=top_k,
        )

        result: DeliberationResult = full_state["final_answer"]

        # Extract conflict edges from serialized graph state
        raw_edges = full_state.get("conflict_graph", {}).get("edges", [])
        conflict_edges = [ConflictEdge(**e) for e in raw_edges]

        return {
            "result": result,
            "conflict_edges": conflict_edges,
            "error": None,
        }
    except Exception as exc:
        log.error("deliberative_example_failed", id=example["id"], error=str(exc))
        return {
            "result": _error_result(example["question"]),
            "conflict_edges": [],
            "error": str(exc),
        }


def run_baseline_example(
    example: dict,
    baseline: BaselineRAG,
) -> dict[str, Any]:
    """Run one example through the baseline RAG pipeline.

    Returns a dict with ``result`` (DeliberationResult) and ``error``
    (str or None).
    """
    try:
        result = baseline.run(example["question"])
        return {"result": result, "error": None}
    except Exception as exc:
        log.error("baseline_example_failed", id=example["id"], error=str(exc))
        return {"result": _error_result(example["question"]), "error": str(exc)}


def _error_result(query: str) -> DeliberationResult:
    """Produce a placeholder result when a pipeline crashes."""
    from src.schemas import ConfidenceLevel

    return DeliberationResult(
        query=query,
        answer="Error during evaluation.",
        confidence=ConfidenceLevel.LOW,
        confidence_score=0.0,
        reasoning_trace=["Pipeline error."],
        source_attribution=[],
        conflict_summary="",
    )


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_name: str = "all",
    max_samples: int | None = None,
    baseline_only: bool = False,
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run the full evaluation loop and return a results dict.

    Args:
        dataset_name: Which benchmark subset to evaluate on.
        max_samples: Cap on number of samples (None = all).
        baseline_only: If True, skip the deliberative system.
        top_k: Number of passages to retrieve per query.

    Returns:
        Nested dict ``{system_name: {metric_name: score}}``.
    """
    examples = load_dataset(dataset_name, max_samples)
    if not examples:
        log.warning("no_examples_found", dataset=dataset_name)
        return {}

    # Initialise shared components
    llm = LLMClient(
        model="claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=2048,
    )
    embedder = EmbeddingModel()
    qdrant = QdrantManager()

    baseline = BaselineRAG(llm, qdrant, embedder, top_k=top_k)

    # Collect predictions
    ground_truths = [ex["ground_truth_answer"] for ex in examples]
    known_conflicts = [ex.get("has_known_conflict", False) for ex in examples]

    # -- Baseline --
    log.info("running_baseline", count=len(examples))
    baseline_preds: list[DeliberationResult] = []
    baseline_errors: list[str | None] = []

    for ex in tqdm(examples, desc="Baseline RAG", unit="query"):
        out = run_baseline_example(ex, baseline)
        baseline_preds.append(out["result"])
        baseline_errors.append(out["error"])

    # -- Deliberative --
    delib_preds: list[DeliberationResult] = []
    delib_conflicts: list[list[ConflictEdge]] = []
    delib_errors: list[str | None] = []

    if not baseline_only:
        log.info("running_deliberative", count=len(examples))
        for ex in tqdm(examples, desc="Deliberative RAG", unit="query"):
            out = run_deliberative_example(ex, llm, qdrant, embedder, top_k)
            delib_preds.append(out["result"])
            delib_conflicts.append(out["conflict_edges"])
            delib_errors.append(out["error"])

    # -- Compute metrics --
    log.info("computing_metrics")

    # Use a separate LLM instance for judging (same model, but distinct
    # for clarity in logs)
    judge_llm = LLMClient(
        model="claude-sonnet-4-6",
        temperature=0.0,
        max_tokens=64,
    )

    results: dict[str, dict[str, Any]] = {}

    # Baseline metrics — judge once, reuse for both accuracy and calibration
    bl_correctness = _compute_correctness(baseline_preds, ground_truths, judge_llm)
    bl_accuracy = sum(bl_correctness) / len(bl_correctness) if bl_correctness else 0.0
    bl_calibration = confidence_calibration(baseline_preds, bl_correctness)
    bl_ece = calibration_error(bl_calibration)

    results["Baseline RAG"] = {
        "factual_accuracy": round(bl_accuracy, 4),
        "conflict_recall": 0.0,  # baseline has no conflict detection
        "calibration_error": bl_ece,
        "calibration": bl_calibration,
        "error_count": sum(1 for e in baseline_errors if e is not None),
    }

    # Deliberative metrics
    if not baseline_only:
        dl_correctness = _compute_correctness(delib_preds, ground_truths, judge_llm)
        dl_accuracy = sum(dl_correctness) / len(dl_correctness) if dl_correctness else 0.0
        dl_conflict_recall = conflict_detection_recall(delib_conflicts, known_conflicts)
        dl_calibration = confidence_calibration(delib_preds, dl_correctness)
        dl_ece = calibration_error(dl_calibration)

        results["Deliberative RAG"] = {
            "factual_accuracy": round(dl_accuracy, 4),
            "conflict_recall": dl_conflict_recall,
            "calibration_error": dl_ece,
            "calibration": dl_calibration,
            "error_count": sum(1 for e in delib_errors if e is not None),
        }

    # Clean up Qdrant
    qdrant.close()

    return results


def _compute_correctness(
    predictions: list[DeliberationResult],
    ground_truths: list[str | list[str]],
    llm: LLMClient,
) -> list[bool]:
    """Return per-prediction boolean correctness via LLM judge.

    This duplicates the judge calls that factual_accuracy makes.  To avoid
    doubling the LLM cost, callers should ideally cache — but for clarity
    in the evaluation runner we keep it simple and call the judge once
    within this function, then reuse the boolean list for both accuracy
    and calibration.
    """
    from evaluation.metrics import _judge_single

    return [
        _judge_single(pred.answer, gt, llm)
        for pred, gt in zip(predictions, ground_truths)
    ]


# ---------------------------------------------------------------------------
# Saving & printing
# ---------------------------------------------------------------------------


def save_results(
    results: dict[str, dict[str, Any]],
    dataset_name: str,
    examples: list[dict] | None = None,
) -> Path:
    """Persist evaluation results as JSON to evaluation/results/.

    Args:
        results: Metrics dict from :func:`run_evaluation`.
        dataset_name: Dataset name used in the filename.
        examples: Optional example list for record-keeping.

    Returns:
        Path to the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{dataset_name}_{timestamp}.json"
    path = RESULTS_DIR / filename

    payload = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "example_count": examples and len(examples),
        "results": _make_serializable(results),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info("results_saved", path=str(path))
    return path


def _make_serializable(obj: Any) -> Any:
    """Convert non-JSON-safe types for serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def print_results_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: Results dict from :func:`run_evaluation`.
    """
    report = summary_report(results)
    print(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    start = time.time()

    results = run_evaluation(
        dataset_name=args.dataset,
        max_samples=args.sample,
        baseline_only=args.baseline_only,
        top_k=args.top_k,
    )

    elapsed = time.time() - start
    log.info("evaluation_complete", elapsed_sec=round(elapsed, 1))

    if results:
        print_results_table(results)
        path = save_results(results, args.dataset)
        print(f"\nResults saved to {path}")
    else:
        print("No results to display.")


if __name__ == "__main__":
    main()
