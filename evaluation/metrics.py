"""Evaluation metric computations.

Implements the three core metrics defined in CLAUDE.md:
1. Factual Accuracy (LLM-as-judge)
2. Conflict Detection Recall
3. Confidence Calibration
"""

from __future__ import annotations

import structlog

from src.schemas import ConfidenceLevel, ConflictEdge, DeliberationResult, RelationType
from src.utils.llm import LLMClient
from src.utils.prompts import load_prompt

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# 1. Factual Accuracy — LLM-as-judge
# ---------------------------------------------------------------------------


def factual_accuracy(
    predictions: list[DeliberationResult],
    ground_truths: list[str | list[str]],
    llm: LLMClient,
) -> float:
    """Compute factual accuracy using an LLM judge.

    For each prediction the judge decides whether the generated answer is
    factually consistent with the ground truth.  When the ground truth is
    a list of acceptable answers, the prediction is CORRECT if the judge
    accepts it against *any* of them.

    Args:
        predictions: System-generated results.
        ground_truths: Reference answers (string or list of acceptable
            strings), same order as *predictions*.
        llm: LLM client used as the judge.

    Returns:
        Accuracy score in [0, 1].  Returns 0.0 when *predictions* is empty.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if _judge_single(pred.answer, gt, llm):
            correct += 1

    accuracy = correct / len(predictions)
    log.info("factual_accuracy", correct=correct, total=len(predictions), accuracy=accuracy)
    return accuracy


def _judge_single(
    prediction: str,
    ground_truth: str | list[str],
    llm: LLMClient,
) -> bool:
    """Ask the LLM judge whether *prediction* matches *ground_truth*.

    Args:
        prediction: System-generated answer text.
        ground_truth: One or more acceptable reference answers.
        llm: LLM client for the judge call.

    Returns:
        True if the judge says CORRECT for any reference answer.
    """
    if isinstance(ground_truth, list):
        gt_text = " OR ".join(ground_truth)
    else:
        gt_text = ground_truth

    prompt = load_prompt(
        "factual_judge",
        ground_truth=gt_text,
        prediction=prediction,
    )
    raw = llm.invoke(prompt).strip().upper()

    is_correct = "CORRECT" in raw and "INCORRECT" not in raw
    log.debug(
        "judge_verdict",
        prediction=prediction[:60],
        verdict=raw[:20],
        is_correct=is_correct,
    )
    return is_correct


# ---------------------------------------------------------------------------
# 2. Conflict Detection Recall
# ---------------------------------------------------------------------------


def conflict_detection_recall(
    system_conflicts: list[list[ConflictEdge]],
    known_conflicts: list[bool],
) -> float:
    """Measure how often the system detects known contradictions.

    Each element in *known_conflicts* indicates whether the corresponding
    query example is known to contain contradicting sources (e.g. from
    ConflictQA).  A conflict is "detected" when the system produced at
    least one ``CONTRADICTS`` edge for that example.

    Args:
        system_conflicts: Per-example lists of ConflictEdge objects
            produced by the system's conflict graph stage.
        known_conflicts: Per-example boolean — True when the example is
            known to contain a contradiction.

    Returns:
        Recall in [0, 1].  Returns 1.0 when there are no known conflicts
        (vacuously true).
    """
    total_known = sum(1 for k in known_conflicts if k)
    if total_known == 0:
        return 1.0

    detected = 0
    for edges, has_conflict in zip(system_conflicts, known_conflicts):
        if not has_conflict:
            continue
        if _has_contradiction(edges):
            detected += 1

    recall = detected / total_known
    log.info("conflict_detection_recall", detected=detected, total_known=total_known, recall=recall)
    return recall


def _has_contradiction(edges: list[ConflictEdge]) -> bool:
    """Check whether any edge is a CONTRADICTS relation."""
    return any(e.relation == RelationType.CONTRADICTS for e in edges)


# ---------------------------------------------------------------------------
# 3. Confidence Calibration
# ---------------------------------------------------------------------------


def confidence_calibration(
    predictions: list[DeliberationResult],
    correctness: list[bool],
) -> dict[str, dict[str, float | int]]:
    """Group predictions by stated confidence and measure actual accuracy.

    Args:
        predictions: System-generated results with confidence levels.
        correctness: Per-prediction boolean indicating factual correctness
            (typically from the LLM judge).

    Returns:
        Dict keyed by confidence level (``"high"``, ``"moderate"``,
        ``"low"``).  Each value is ``{"accuracy": float, "count": int}``.
        A level with zero predictions gets ``accuracy: 0.0, count: 0``.
    """
    buckets: dict[str, list[bool]] = {
        ConfidenceLevel.HIGH: [],
        ConfidenceLevel.MODERATE: [],
        ConfidenceLevel.LOW: [],
    }

    for pred, is_correct in zip(predictions, correctness):
        buckets[pred.confidence.value].append(is_correct)

    result: dict[str, dict[str, float | int]] = {}
    for level in (ConfidenceLevel.HIGH, ConfidenceLevel.MODERATE, ConfidenceLevel.LOW):
        entries = buckets[level.value]
        count = len(entries)
        acc = sum(entries) / count if count > 0 else 0.0
        result[level.value] = {"accuracy": round(acc, 4), "count": count}

    log.info("confidence_calibration", buckets={k: v["count"] for k, v in result.items()})
    return result


def calibration_error(calibration: dict[str, dict[str, float | int]]) -> float:
    """Compute a weighted Expected Calibration Error (ECE).

    Target accuracy per bucket: high → 0.9, moderate → 0.6, low → 0.3.

    Args:
        calibration: Output of :func:`confidence_calibration`.

    Returns:
        ECE score in [0, 1].  Lower is better (0 = perfect calibration).
    """
    targets = {"high": 0.9, "moderate": 0.6, "low": 0.3}
    total_count = sum(v["count"] for v in calibration.values())

    if total_count == 0:
        return 0.0

    ece = 0.0
    for level, target in targets.items():
        entry = calibration.get(level, {"accuracy": 0.0, "count": 0})
        weight = entry["count"] / total_count
        ece += weight * abs(entry["accuracy"] - target)

    return round(ece, 4)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def summary_report(
    all_metrics: dict[str, dict[str, object]],
) -> str:
    """Format a comparison table between systems.

    Args:
        all_metrics: Nested dict ``{system_name: {metric_name: value}}``.
            Expected metrics: ``"factual_accuracy"``, ``"conflict_recall"``,
            ``"calibration_error"``.

    Returns:
        Formatted multi-line string suitable for printing.
    """
    systems = list(all_metrics.keys())

    # Header
    col_width = max(22, *(len(s) for s in systems)) + 2
    metric_names = ["factual_accuracy", "conflict_recall", "calibration_error"]
    metric_labels = ["Factual Accuracy", "Conflict Recall", "Calibration Error (ECE)"]

    header = f"{'Metric':<26}" + "".join(f"{s:>{col_width}}" for s in systems)
    separator = "-" * len(header)

    lines = [
        "",
        "Evaluation Results",
        "=" * len(header),
        header,
        separator,
    ]

    for name, label in zip(metric_names, metric_labels):
        row = f"{label:<26}"
        for system in systems:
            val = all_metrics[system].get(name, "N/A")
            if isinstance(val, float):
                row += f"{val:>{col_width}.4f}"
            else:
                row += f"{str(val):>{col_width}}"
        lines.append(row)

    # Calibration detail rows
    lines.append(separator)
    lines.append("Calibration Breakdown:")
    for level in ("high", "moderate", "low"):
        row = f"  {level:<24}"
        for system in systems:
            cal = all_metrics[system].get("calibration", {})
            entry = cal.get(level, {})
            acc = entry.get("accuracy", 0.0)
            count = entry.get("count", 0)
            cell = f"{acc:.0%} ({count})"
            row += f"{cell:>{col_width}}"
        lines.append(row)

    lines.append(separator)
    lines.append("")
    return "\n".join(lines)
