"""Tests for evaluation/metrics.py — factual accuracy, conflict recall, calibration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluation.metrics import (
    _has_contradiction,
    _judge_single,
    calibration_error,
    confidence_calibration,
    conflict_detection_recall,
    factual_accuracy,
    summary_report,
)
from src.schemas import (
    ConfidenceLevel,
    ConflictEdge,
    DeliberationResult,
    RelationType,
    SourceAttribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    answer: str = "test",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    score: float = 0.9,
) -> DeliberationResult:
    return DeliberationResult(
        query="q",
        answer=answer,
        confidence=confidence,
        confidence_score=score,
        reasoning_trace=["step"],
        source_attribution=[],
        conflict_summary="",
    )


def _edge(relation: str = "CONTRADICTS") -> ConflictEdge:
    return ConflictEdge(
        source_claim_id="c1",
        target_claim_id="c2",
        relation=RelationType(relation),
        confidence=0.9,
        reasoning="test",
    )


def _mock_judge_llm(verdicts: list[str]) -> MagicMock:
    """Return a mock LLM that returns verdicts in order."""
    llm = MagicMock()
    llm.invoke.side_effect = verdicts
    return llm


# ===========================================================================
# 1. Factual Accuracy
# ===========================================================================


class TestFactualAccuracy:

    def test_all_correct(self) -> None:
        preds = [_result("Paris"), _result("Berlin")]
        gts = ["Paris", "Berlin"]
        llm = _mock_judge_llm(["CORRECT", "CORRECT"])
        assert factual_accuracy(preds, gts, llm) == 1.0

    def test_all_incorrect(self) -> None:
        preds = [_result("wrong"), _result("wrong")]
        gts = ["right", "right"]
        llm = _mock_judge_llm(["INCORRECT", "INCORRECT"])
        assert factual_accuracy(preds, gts, llm) == 0.0

    def test_mixed(self) -> None:
        preds = [_result("a"), _result("b"), _result("c"), _result("d")]
        gts = ["a", "b", "c", "d"]
        llm = _mock_judge_llm(["CORRECT", "INCORRECT", "CORRECT", "INCORRECT"])
        assert factual_accuracy(preds, gts, llm) == 0.5

    def test_empty_predictions(self) -> None:
        llm = MagicMock()
        assert factual_accuracy([], [], llm) == 0.0
        llm.invoke.assert_not_called()

    def test_list_ground_truth_joins_with_or(self) -> None:
        llm = _mock_judge_llm(["CORRECT"])
        preds = [_result("politician")]
        gts = [["politician", "political leader"]]
        assert factual_accuracy(preds, gts, llm) == 1.0
        prompt_sent = llm.invoke.call_args[0][0]
        assert "politician OR political leader" in prompt_sent


class TestJudgeSingle:

    def test_correct_verdict(self) -> None:
        llm = _mock_judge_llm(["CORRECT"])
        assert _judge_single("Paris", "Paris", llm) is True

    def test_incorrect_verdict(self) -> None:
        llm = _mock_judge_llm(["INCORRECT"])
        assert _judge_single("wrong", "right", llm) is False

    def test_correct_with_extra_text(self) -> None:
        llm = _mock_judge_llm(["CORRECT."])
        assert _judge_single("a", "a", llm) is True

    def test_incorrect_trumps_correct_substring(self) -> None:
        """'INCORRECT' contains 'CORRECT' — must NOT count as correct."""
        llm = _mock_judge_llm(["INCORRECT"])
        assert _judge_single("a", "b", llm) is False

    def test_case_insensitive(self) -> None:
        llm = _mock_judge_llm(["correct"])
        assert _judge_single("a", "a", llm) is True


# ===========================================================================
# 2. Conflict Detection Recall
# ===========================================================================


class TestConflictDetectionRecall:

    def test_all_detected(self) -> None:
        edges = [[_edge("CONTRADICTS")], [_edge("CONTRADICTS")]]
        known = [True, True]
        assert conflict_detection_recall(edges, known) == 1.0

    def test_none_detected(self) -> None:
        edges = [[_edge("SUPPORTS")], []]
        known = [True, True]
        assert conflict_detection_recall(edges, known) == 0.0

    def test_partial_detection(self) -> None:
        edges = [[_edge("CONTRADICTS")], [_edge("SUPPORTS")]]
        known = [True, True]
        assert conflict_detection_recall(edges, known) == 0.5

    def test_no_known_conflicts_returns_1(self) -> None:
        edges = [[], []]
        known = [False, False]
        assert conflict_detection_recall(edges, known) == 1.0

    def test_ignores_non_conflict_examples(self) -> None:
        """Examples where has_conflict=False should not affect recall."""
        edges = [
            [_edge("CONTRADICTS")],   # known conflict, detected
            [_edge("SUPPORTS")],       # NOT a known conflict — ignored
        ]
        known = [True, False]
        assert conflict_detection_recall(edges, known) == 1.0

    def test_supersedes_is_not_contradicts(self) -> None:
        """Only CONTRADICTS counts as a detected conflict."""
        edges = [[_edge("SUPERSEDES")]]
        known = [True]
        assert conflict_detection_recall(edges, known) == 0.0

    def test_multiple_edges_one_contradicts_suffices(self) -> None:
        edges = [[_edge("SUPPORTS"), _edge("CONTRADICTS")]]
        known = [True]
        assert conflict_detection_recall(edges, known) == 1.0


class TestHasContradiction:

    def test_empty_edges(self) -> None:
        assert _has_contradiction([]) is False

    def test_only_supports(self) -> None:
        assert _has_contradiction([_edge("SUPPORTS")]) is False

    def test_has_contradicts(self) -> None:
        assert _has_contradiction([_edge("CONTRADICTS")]) is True


# ===========================================================================
# 3. Confidence Calibration
# ===========================================================================


class TestConfidenceCalibration:

    def test_all_high_all_correct(self) -> None:
        preds = [_result(confidence=ConfidenceLevel.HIGH)] * 3
        correctness = [True, True, True]
        cal = confidence_calibration(preds, correctness)
        assert cal["high"]["accuracy"] == 1.0
        assert cal["high"]["count"] == 3
        assert cal["moderate"]["count"] == 0
        assert cal["low"]["count"] == 0

    def test_mixed_confidence_levels(self) -> None:
        preds = [
            _result(confidence=ConfidenceLevel.HIGH),
            _result(confidence=ConfidenceLevel.HIGH),
            _result(confidence=ConfidenceLevel.MODERATE),
            _result(confidence=ConfidenceLevel.LOW),
        ]
        correctness = [True, False, True, False]
        cal = confidence_calibration(preds, correctness)
        assert cal["high"]["accuracy"] == 0.5
        assert cal["high"]["count"] == 2
        assert cal["moderate"]["accuracy"] == 1.0
        assert cal["moderate"]["count"] == 1
        assert cal["low"]["accuracy"] == 0.0
        assert cal["low"]["count"] == 1

    def test_empty_input(self) -> None:
        cal = confidence_calibration([], [])
        assert cal["high"]["count"] == 0
        assert cal["moderate"]["count"] == 0
        assert cal["low"]["count"] == 0

    def test_all_levels_present(self) -> None:
        """Result should always have all three levels, even if count is 0."""
        preds = [_result(confidence=ConfidenceLevel.MODERATE)]
        cal = confidence_calibration(preds, [True])
        assert "high" in cal
        assert "moderate" in cal
        assert "low" in cal


class TestCalibrationError:

    def test_perfect_calibration(self) -> None:
        """Targets: high=0.9, moderate=0.6, low=0.3."""
        cal = {
            "high": {"accuracy": 0.9, "count": 10},
            "moderate": {"accuracy": 0.6, "count": 10},
            "low": {"accuracy": 0.3, "count": 10},
        }
        assert calibration_error(cal) == 0.0

    def test_worst_case_high_only(self) -> None:
        """All high-confidence, none correct → accuracy=0, target=0.9, ECE=0.9."""
        cal = {
            "high": {"accuracy": 0.0, "count": 10},
            "moderate": {"accuracy": 0.0, "count": 0},
            "low": {"accuracy": 0.0, "count": 0},
        }
        assert calibration_error(cal) == pytest.approx(0.9)

    def test_empty(self) -> None:
        cal = {
            "high": {"accuracy": 0.0, "count": 0},
            "moderate": {"accuracy": 0.0, "count": 0},
            "low": {"accuracy": 0.0, "count": 0},
        }
        assert calibration_error(cal) == 0.0

    def test_weighted_by_count(self) -> None:
        """10 high (acc=0.5) + 10 low (acc=0.3): ECE = 0.5*|0.5-0.9| + 0.5*|0.3-0.3| = 0.2."""
        cal = {
            "high": {"accuracy": 0.5, "count": 10},
            "moderate": {"accuracy": 0.0, "count": 0},
            "low": {"accuracy": 0.3, "count": 10},
        }
        assert calibration_error(cal) == pytest.approx(0.2)


# ===========================================================================
# Summary report
# ===========================================================================


class TestSummaryReport:

    def test_contains_all_systems(self) -> None:
        metrics = {
            "deliberative": {
                "factual_accuracy": 0.85,
                "conflict_recall": 0.72,
                "calibration_error": 0.08,
                "calibration": {
                    "high": {"accuracy": 0.9, "count": 10},
                    "moderate": {"accuracy": 0.6, "count": 5},
                    "low": {"accuracy": 0.3, "count": 2},
                },
            },
            "baseline": {
                "factual_accuracy": 0.65,
                "conflict_recall": 0.0,
                "calibration_error": 0.35,
                "calibration": {
                    "high": {"accuracy": 0.65, "count": 17},
                    "moderate": {"accuracy": 0.0, "count": 0},
                    "low": {"accuracy": 0.0, "count": 0},
                },
            },
        }
        report = summary_report(metrics)
        assert "deliberative" in report
        assert "baseline" in report

    def test_contains_metric_labels(self) -> None:
        metrics = {
            "system_a": {
                "factual_accuracy": 0.8,
                "conflict_recall": 0.7,
                "calibration_error": 0.1,
                "calibration": {
                    "high": {"accuracy": 0.9, "count": 5},
                    "moderate": {"accuracy": 0.6, "count": 3},
                    "low": {"accuracy": 0.3, "count": 2},
                },
            },
        }
        report = summary_report(metrics)
        assert "Factual Accuracy" in report
        assert "Conflict Recall" in report
        assert "Calibration Error" in report

    def test_returns_string(self) -> None:
        metrics = {"sys": {
            "factual_accuracy": 0.5,
            "conflict_recall": 0.5,
            "calibration_error": 0.5,
            "calibration": {
                "high": {"accuracy": 0.5, "count": 1},
                "moderate": {"accuracy": 0.0, "count": 0},
                "low": {"accuracy": 0.0, "count": 0},
            },
        }}
        assert isinstance(summary_report(metrics), str)

    def test_calibration_breakdown_rows(self) -> None:
        metrics = {"sys": {
            "factual_accuracy": 0.5,
            "conflict_recall": 0.5,
            "calibration_error": 0.5,
            "calibration": {
                "high": {"accuracy": 0.85, "count": 10},
                "moderate": {"accuracy": 0.55, "count": 5},
                "low": {"accuracy": 0.2, "count": 3},
            },
        }}
        report = summary_report(metrics)
        assert "high" in report
        assert "moderate" in report
        assert "low" in report
