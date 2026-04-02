"""Evaluation runner script.

Evaluates both the deliberative RAG system and the baseline RAG pipeline
across all three benchmark datasets and writes results to evaluation/results/.

Usage:
    python -m evaluation.run_eval [--dataset conflict_qa|frames|finance_bench]
                                   [--max-samples N]
                                   [--baseline-only]
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script.

    Returns:
        Parsed argument namespace.
    """
    pass


def run_evaluation(
    dataset_name: str,
    max_samples: int | None,
    baseline_only: bool,
) -> dict[str, dict[str, float]]:
    """Run the full evaluation loop and return a results dict.

    Args:
        dataset_name: Which benchmark to evaluate on.
        max_samples: Cap on number of samples (None = all).
        baseline_only: If True, skip the deliberative system.

    Returns:
        Nested dict: ``{system_name: {metric_name: score}}``.
    """
    pass


def save_results(results: dict[str, dict[str, float]], dataset_name: str) -> None:
    """Persist evaluation results as JSON to evaluation/results/.

    Args:
        results: Results dict from run_evaluation.
        dataset_name: Dataset name used as part of the filename.
    """
    pass


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: Results dict from run_evaluation.
    """
    pass


if __name__ == "__main__":
    args = parse_args()
    results = run_evaluation(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        baseline_only=args.baseline_only,
    )
    print_results_table(results)
    save_results(results, args.dataset)
