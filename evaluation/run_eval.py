"""Evaluation runner script.

Evaluates both the deliberative RAG system and the baseline RAG pipeline
across the master evaluation dataset and writes results to
``evaluation/results/``.

Usage::

    python -m evaluation.run_eval                     # full run (200 examples)
    python -m evaluation.run_eval --sample 10         # quick smoke-test
    python -m evaluation.run_eval --dataset conflict_qa --sample 20
    python -m evaluation.run_eval --baseline-only --sample 5
    python -m evaluation.run_eval --ablation --sample 10   # ablation study
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

from evaluation.ablations import (
    ABLATION_CONFIGS,
    AblationRunner,
    format_ablation_table,
)
from evaluation.baseline_rag import BaselineRAG
from evaluation.metrics import (
    calibration_error,
    confidence_calibration,
    conflict_detection_recall,
    summary_report,
)
from src.agent.graph import run_query_full
from src.schemas import ConflictEdge, DeliberationResult
from src.utils.llm import LLMClient, get_llm
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
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study (full system + 4 variants + baseline).",
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
    llm_heavy: LLMClient,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 5,
    llm_light: LLMClient | None = None,
) -> dict[str, Any]:
    """Run one example through the deliberative RAG pipeline.

    Returns a dict with ``result`` (DeliberationResult), ``conflict_edges``
    (list of ConflictEdge), and ``error`` (str or None).
    """
    try:
        full_state = run_query_full(
            example["question"],
            llm_heavy=llm_heavy,
            qdrant=qdrant,
            embedder=embedder,
            top_k=top_k,
            llm_light=llm_light,
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
# Ablation study
# ---------------------------------------------------------------------------


def run_ablation_study(
    dataset_name: str = "all",
    max_samples: int | None = None,
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run the full system, all 4 ablation variants, and the baseline.

    Args:
        dataset_name: Which benchmark subset to evaluate on.
        max_samples: Cap on number of examples.
        top_k: Passages per query.

    Returns:
        Ordered dict ``{config_label: {metric: value}}``.
    """
    examples = load_dataset(dataset_name, max_samples)
    if not examples:
        log.warning("no_examples_found", dataset=dataset_name)
        return {}

    llm_heavy = get_llm("heavy")
    llm_light = get_llm("light")
    judge_llm = get_llm("heavy")
    embedder = EmbeddingModel()
    qdrant = QdrantManager()

    ground_truths = [ex["ground_truth_answer"] for ex in examples]
    known_conflicts = [ex.get("has_known_conflict", False) for ex in examples]

    results: dict[str, dict[str, Any]] = {}

    # --- Full system (no ablation) ---
    log.info("ablation_full_system", count=len(examples))
    full_preds, full_conflicts = _run_system_on_examples(
        examples, llm_heavy, qdrant, embedder, top_k,
        label="Full System", llm_light=llm_light,
    )
    results["Full System"] = _compute_system_metrics(
        full_preds, full_conflicts, ground_truths, known_conflicts, judge_llm,
    )

    # --- 4 ablation variants ---
    for variant_key in ["no_temporal", "no_authority", "no_graph", "no_claims"]:
        config = ABLATION_CONFIGS[variant_key]
        runner = AblationRunner(llm_heavy, qdrant, embedder, config, top_k=top_k)
        label = f"- {config.name}"

        log.info("ablation_variant_start", variant=config.name, count=len(examples))
        preds: list[DeliberationResult] = []
        conflicts: list[list[ConflictEdge]] = []

        for ex in tqdm(examples, desc=label, unit="query"):
            out = runner.run(ex["question"])
            preds.append(out["result"])
            conflicts.append(out["conflict_edges"])

        results[label] = _compute_system_metrics(
            preds, conflicts, ground_truths, known_conflicts, judge_llm,
        )

    # --- Baseline ---
    log.info("ablation_baseline", count=len(examples))
    baseline = BaselineRAG(llm_heavy, qdrant, embedder, top_k=top_k)
    bl_preds: list[DeliberationResult] = []
    for ex in tqdm(examples, desc="Baseline RAG", unit="query"):
        out = run_baseline_example(ex, baseline)
        bl_preds.append(out["result"])

    bl_correctness = _compute_correctness(bl_preds, ground_truths, judge_llm)
    bl_accuracy = sum(bl_correctness) / len(bl_correctness) if bl_correctness else 0.0
    bl_calibration = confidence_calibration(bl_preds, bl_correctness)
    results["Baseline RAG"] = {
        "factual_accuracy": round(bl_accuracy, 4),
        "conflict_recall": 0.0,
        "calibration_error": calibration_error(bl_calibration),
        "calibration": bl_calibration,
    }

    qdrant.close()
    return results


def _run_system_on_examples(
    examples: list[dict],
    llm_heavy: LLMClient,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int,
    label: str = "Deliberative",
    llm_light: LLMClient | None = None,
) -> tuple[list[DeliberationResult], list[list[ConflictEdge]]]:
    """Run the full deliberative pipeline on a list of examples."""
    preds: list[DeliberationResult] = []
    conflicts: list[list[ConflictEdge]] = []
    for ex in tqdm(examples, desc=label, unit="query"):
        out = run_deliberative_example(
            ex, llm_heavy, qdrant, embedder, top_k, llm_light=llm_light,
        )
        preds.append(out["result"])
        conflicts.append(out["conflict_edges"])
    return preds, conflicts


def _compute_system_metrics(
    preds: list[DeliberationResult],
    conflicts: list[list[ConflictEdge]],
    ground_truths: list[str | list[str]],
    known_conflicts: list[bool],
    judge_llm: LLMClient,
) -> dict[str, Any]:
    """Compute all three metrics for a system's predictions."""
    correctness = _compute_correctness(preds, ground_truths, judge_llm)
    accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    cr = conflict_detection_recall(conflicts, known_conflicts)
    cal = confidence_calibration(preds, correctness)
    ece = calibration_error(cal)
    return {
        "factual_accuracy": round(accuracy, 4),
        "conflict_recall": cr,
        "calibration_error": ece,
        "calibration": cal,
    }


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
    llm_heavy = get_llm("heavy")
    llm_light = get_llm("light")
    judge_llm = get_llm("heavy")
    embedder = EmbeddingModel()
    qdrant = QdrantManager()

    baseline = BaselineRAG(llm_heavy, qdrant, embedder, top_k=top_k)

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
            out = run_deliberative_example(
                ex, llm_heavy, qdrant, embedder, top_k, llm_light=llm_light,
            )
            delib_preds.append(out["result"])
            delib_conflicts.append(out["conflict_edges"])
            delib_errors.append(out["error"])

    # -- Compute metrics --
    log.info("computing_metrics")

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

    if args.ablation:
        results = run_ablation_study(
            dataset_name=args.dataset,
            max_samples=args.sample,
            top_k=args.top_k,
        )
        elapsed = time.time() - start
        log.info("ablation_complete", elapsed_sec=round(elapsed, 1))

        if results:
            table = format_ablation_table(results)
            print(table)
            path = save_results(results, f"ablation_{args.dataset}")
            print(f"\nResults saved to {path}")
        else:
            print("No results to display.")
    else:
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
