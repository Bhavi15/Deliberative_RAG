"""Evaluation metric computations.

Implements the three core metrics defined in CLAUDE.md:
1. Factual Accuracy
2. Conflict Detection Recall
3. Confidence Calibration (ECE)
"""

from src.schemas import Answer


def compute_factual_accuracy(predictions: list[Answer], ground_truths: list[str]) -> float:
    """Compute factual accuracy as the fraction of answers matching ground truth.

    Args:
        predictions: Model-generated Answer objects.
        ground_truths: Reference answer strings (same order).

    Returns:
        Accuracy score in [0, 1].
    """
    pass


def compute_conflict_recall(
    predicted_conflicts: list[list[str]],
    known_conflicts: list[list[str]],
) -> float:
    """Compute what fraction of known contradictions the system detected.

    Args:
        predicted_conflicts: Per-query lists of detected conflict claim-ID pairs.
        known_conflicts: Per-query lists of ground-truth conflict pairs.

    Returns:
        Recall score in [0, 1].
    """
    pass


def compute_calibration(predictions: list[Answer], ground_truths: list[str]) -> float:
    """Measure calibration via Expected Calibration Error (ECE).

    Args:
        predictions: Model-generated answers with confidence scores.
        ground_truths: Reference answer strings.

    Returns:
        ECE score; lower is better (0 = perfect calibration).
    """
    pass


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Case-insensitive, whitespace-normalised exact string match.

    Args:
        prediction: Predicted answer string.
        ground_truth: Reference answer string.

    Returns:
        True if strings match after normalisation.
    """
    pass
