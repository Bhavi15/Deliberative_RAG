"""Tests for evaluation/run_eval.py — dataset loading, example runners, metric wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evaluation.run_eval import (
    _compute_correctness,
    _error_result,
    load_dataset,
    run_baseline_example,
    run_deliberative_example,
    save_results,
)
from src.schemas import ConfidenceLevel, DeliberationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATASET = [
    {
        "id": "cq_001",
        "source_dataset": "conflict_qa",
        "question": "What is X's occupation?",
        "ground_truth_answer": ["politician"],
        "evidence_passages": [{"text": "X is a musician."}, {"text": "X is a politician."}],
        "has_known_conflict": True,
        "metadata": {},
    },
    {
        "id": "fb_001",
        "source_dataset": "finance_bench",
        "question": "What was revenue?",
        "ground_truth_answer": "$32.8B",
        "evidence_passages": [{"text": "Revenue was $32.8 billion."}],
        "has_known_conflict": False,
        "metadata": {},
    },
    {
        "id": "fr_001",
        "source_dataset": "frames",
        "question": "Who wrote it?",
        "ground_truth_answer": "Jane Doe",
        "evidence_passages": [{"text": "Jane Doe wrote the book."}],
        "has_known_conflict": False,
        "metadata": {},
    },
]


@pytest.fixture
def dataset_file(tmp_path: Path) -> Path:
    p = tmp_path / "master_eval_dataset.json"
    p.write_text(json.dumps(SAMPLE_DATASET), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestLoadDataset:

    def test_loads_all(self, dataset_file: Path) -> None:
        with patch("evaluation.run_eval.DATASET_PATH", dataset_file):
            data = load_dataset("all")
        assert len(data) == 3

    def test_filters_by_dataset(self, dataset_file: Path) -> None:
        with patch("evaluation.run_eval.DATASET_PATH", dataset_file):
            data = load_dataset("conflict_qa")
        assert len(data) == 1
        assert data[0]["id"] == "cq_001"

    def test_sample_cap(self, dataset_file: Path) -> None:
        with patch("evaluation.run_eval.DATASET_PATH", dataset_file):
            data = load_dataset("all", max_samples=2)
        assert len(data) == 2

    def test_sample_larger_than_data(self, dataset_file: Path) -> None:
        with patch("evaluation.run_eval.DATASET_PATH", dataset_file):
            data = load_dataset("all", max_samples=100)
        assert len(data) == 3


# ---------------------------------------------------------------------------
# Example runners
# ---------------------------------------------------------------------------


class TestRunBaselineExample:

    def test_returns_result(self) -> None:
        baseline = MagicMock()
        baseline.run.return_value = DeliberationResult(
            query="q", answer="a", confidence=ConfidenceLevel.HIGH,
            confidence_score=1.0, reasoning_trace=[], source_attribution=[],
            conflict_summary="",
        )
        out = run_baseline_example(SAMPLE_DATASET[0], baseline)
        assert isinstance(out["result"], DeliberationResult)
        assert out["error"] is None

    def test_catches_exception(self) -> None:
        baseline = MagicMock()
        baseline.run.side_effect = RuntimeError("boom")
        out = run_baseline_example(SAMPLE_DATASET[0], baseline)
        assert out["error"] is not None
        assert out["result"].confidence == ConfidenceLevel.LOW


class TestRunDeliberativeExample:

    def test_returns_result_and_edges(self) -> None:
        mock_state = {
            "final_answer": DeliberationResult(
                query="q", answer="a", confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9, reasoning_trace=["s1"], source_attribution=[],
                conflict_summary="c",
            ),
            "conflict_graph": {
                "nodes": ["c1", "c2"],
                "edges": [{
                    "source_claim_id": "c1",
                    "target_claim_id": "c2",
                    "relation": "CONTRADICTS",
                    "confidence": 0.9,
                    "reasoning": "test",
                }],
            },
        }
        with patch("evaluation.run_eval.run_query_full", return_value=mock_state):
            out = run_deliberative_example(
                SAMPLE_DATASET[0],
                llm_heavy=MagicMock(), qdrant=MagicMock(), embedder=MagicMock(),
            )
        assert isinstance(out["result"], DeliberationResult)
        assert len(out["conflict_edges"]) == 1
        assert out["conflict_edges"][0].relation.value == "CONTRADICTS"
        assert out["error"] is None

    def test_catches_exception(self) -> None:
        with patch("evaluation.run_eval.run_query_full", side_effect=RuntimeError("fail")):
            out = run_deliberative_example(
                SAMPLE_DATASET[0],
                llm_heavy=MagicMock(), qdrant=MagicMock(), embedder=MagicMock(),
            )
        assert out["error"] is not None
        assert out["result"].confidence == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestErrorResult:

    def test_produces_low_confidence(self) -> None:
        r = _error_result("my query")
        assert r.confidence == ConfidenceLevel.LOW
        assert r.confidence_score == 0.0
        assert r.query == "my query"


class TestComputeCorrectness:

    def test_returns_booleans(self) -> None:
        preds = [
            DeliberationResult(
                query="q", answer="Paris", confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9, reasoning_trace=[], source_attribution=[],
                conflict_summary="",
            ),
        ]
        llm = MagicMock()
        llm.invoke.return_value = "CORRECT"
        result = _compute_correctness(preds, ["Paris"], llm)
        assert result == [True]

    def test_incorrect(self) -> None:
        preds = [
            DeliberationResult(
                query="q", answer="wrong", confidence=ConfidenceLevel.HIGH,
                confidence_score=0.9, reasoning_trace=[], source_attribution=[],
                conflict_summary="",
            ),
        ]
        llm = MagicMock()
        llm.invoke.return_value = "INCORRECT"
        result = _compute_correctness(preds, ["right"], llm)
        assert result == [False]


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


class TestSaveResults:

    def test_writes_json_file(self, tmp_path: Path) -> None:
        with patch("evaluation.run_eval.RESULTS_DIR", tmp_path):
            path = save_results(
                {"Baseline RAG": {"factual_accuracy": 0.7}},
                "conflict_qa",
            )
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["dataset"] == "conflict_qa"
        assert "Baseline RAG" in data["results"]

    def test_creates_results_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub" / "results"
        with patch("evaluation.run_eval.RESULTS_DIR", nested):
            path = save_results({"sys": {}}, "all")
        assert nested.exists()
        assert path.exists()
