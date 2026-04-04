"""Tests for the BaselineRAG vanilla pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluation.baseline_rag import BaselineRAG
from src.schemas import ConfidenceLevel, DeliberationResult


# -- Helpers ------------------------------------------------------------------

def _mock_qdrant(passages: list[dict] | None = None) -> MagicMock:
    if passages is None:
        passages = [
            {"id": "doc_1", "score": 0.9, "text": "Paris is the capital of France."},
            {"id": "doc_2", "score": 0.8, "text": "France is a country in Western Europe."},
            {"id": "doc_3", "score": 0.7, "text": "The Eiffel Tower is in Paris."},
        ]
    qdrant = MagicMock()
    qdrant.search.return_value = passages
    return qdrant


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 384
    return embedder


def _mock_llm(answer: str = "Paris is the capital of France.") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = answer
    return llm


# -- Tests --------------------------------------------------------------------


class TestBaselineRAG:

    def test_returns_deliberation_result(self) -> None:
        baseline = BaselineRAG(_mock_llm(), _mock_qdrant(), _mock_embedder())
        result = baseline.run("What is the capital of France?")
        assert isinstance(result, DeliberationResult)

    def test_answer_comes_from_llm(self) -> None:
        baseline = BaselineRAG(_mock_llm("Test answer"), _mock_qdrant(), _mock_embedder())
        result = baseline.run("query?")
        assert result.answer == "Test answer"

    def test_confidence_always_high(self) -> None:
        baseline = BaselineRAG(_mock_llm(), _mock_qdrant(), _mock_embedder())
        result = baseline.run("query?")
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.confidence_score == 1.0

    def test_reasoning_trace_empty(self) -> None:
        baseline = BaselineRAG(_mock_llm(), _mock_qdrant(), _mock_embedder())
        result = baseline.run("query?")
        assert result.reasoning_trace == []

    def test_conflict_summary_empty(self) -> None:
        baseline = BaselineRAG(_mock_llm(), _mock_qdrant(), _mock_embedder())
        result = baseline.run("query?")
        assert result.conflict_summary == ""

    def test_source_attribution_matches_passages(self) -> None:
        passages = [
            {"id": "d1", "score": 0.9, "text": "Passage one."},
            {"id": "d2", "score": 0.8, "text": "Passage two."},
        ]
        baseline = BaselineRAG(_mock_llm(), _mock_qdrant(passages), _mock_embedder())
        result = baseline.run("query?")
        assert len(result.source_attribution) == 2
        assert result.source_attribution[0].source_doc_id == "d1"
        assert result.source_attribution[1].source_doc_id == "d2"

    def test_query_preserved(self) -> None:
        baseline = BaselineRAG(_mock_llm(), _mock_qdrant(), _mock_embedder())
        result = baseline.run("What is the capital of France?")
        assert result.query == "What is the capital of France?"

    def test_prompt_contains_all_passages(self) -> None:
        llm = _mock_llm()
        passages = [
            {"id": "d1", "score": 0.9, "text": "Alpha passage."},
            {"id": "d2", "score": 0.8, "text": "Beta passage."},
        ]
        baseline = BaselineRAG(llm, _mock_qdrant(passages), _mock_embedder())
        baseline.run("query?")

        prompt_sent = llm.invoke.call_args[0][0]
        assert "Alpha passage." in prompt_sent
        assert "Beta passage." in prompt_sent
        assert "query?" in prompt_sent

    def test_custom_top_k(self) -> None:
        qdrant = _mock_qdrant()
        baseline = BaselineRAG(_mock_llm(), qdrant, _mock_embedder(), top_k=3)
        baseline.run("query?")
        qdrant.search.assert_called_once()
        assert qdrant.search.call_args[1]["top_k"] == 3

    def test_no_passages_returns_fallback(self) -> None:
        llm = _mock_llm()
        baseline = BaselineRAG(llm, _mock_qdrant(passages=[]), _mock_embedder())
        result = baseline.run("query?")
        assert "don't have enough information" in result.answer
        llm.invoke.assert_not_called()

    def test_strips_whitespace_from_answer(self) -> None:
        baseline = BaselineRAG(_mock_llm("  padded answer  "), _mock_qdrant(), _mock_embedder())
        result = baseline.run("query?")
        assert result.answer == "padded answer"
