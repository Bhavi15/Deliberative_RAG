"""Stage 6 — Deliberation Synthesis.

Builds a scored-claim context string, calls the LLM with the answer
synthesis prompt, and parses the output into a DeliberationResult.
"""

from __future__ import annotations

import json
import re
import textwrap

import networkx as nx
import structlog

from src.schemas import (
    ConfidenceLevel,
    ConflictEdge,
    DeliberationResult,
    RelationType,
    ScoredClaim,
    SourceAttribution,
)
from src.utils.llm import LLMClient
from src.utils.prompts import load_prompt

log = structlog.get_logger()

_VALID_CONFIDENCE = {c.value for c in ConfidenceLevel}


class DeliberationSynthesizer:
    """Produces a final, source-attributed answer from scored claims."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialise the synthesizer.

        Args:
            llm: Configured LLMClient for calling Claude.
        """
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        scored_claims: list[ScoredClaim],
        graph: nx.DiGraph,
    ) -> DeliberationResult:
        """Run the full synthesis pipeline.

        Args:
            query: Original user query.
            scored_claims: Claims ranked by ``final_score`` descending.
            graph: Conflict graph for extracting conflict edges.

        Returns:
            A fully populated DeliberationResult.
        """
        conflicts = _extract_conflicts(graph)
        context = build_claims_context(scored_claims, conflicts)
        conflict_text = build_conflict_summary(conflicts, scored_claims)

        prompt = load_prompt(
            "answer_synthesis",
            query=query,
            scored_claims=context,
            conflict_summary=conflict_text if conflict_text else "No conflicts detected.",
        )

        raw = self._llm.invoke(prompt)
        result = parse_synthesis_response(raw, query)

        if result is not None:
            return result

        # Retry once
        log.warning("synthesis_retry", query=query[:60])
        raw_retry = self._llm.invoke(prompt)
        result_retry = parse_synthesis_response(raw_retry, query)

        if result_retry is not None:
            return result_retry

        # Fallback: build a minimal result from top claim
        log.error("synthesis_failed", query=query[:60])
        return _fallback_result(query, scored_claims)

    @staticmethod
    def format_for_display(result: DeliberationResult) -> str:
        """Render a DeliberationResult as human-readable text.

        Args:
            result: The result to format.

        Returns:
            Formatted multi-line string.
        """
        lines: list[str] = []
        lines.append(f"Query: {result.query}")
        lines.append("")
        lines.append(f"Answer: {result.answer}")
        lines.append(f"Confidence: {result.confidence.value} ({result.confidence_score:.2f})")
        lines.append("")

        if result.reasoning_trace:
            lines.append("Reasoning:")
            for i, step in enumerate(result.reasoning_trace, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")

        if result.source_attribution:
            lines.append("Sources:")
            for sa in result.source_attribution:
                lines.append(
                    f"  - [{sa.claim_id}] from {sa.source_doc_id} "
                    f"(relevance: {sa.relevance:.2f})"
                )
            lines.append("")

        if result.conflict_summary:
            lines.append(f"Conflicts: {result.conflict_summary}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def build_claims_context(
    scored_claims: list[ScoredClaim],
    conflicts: list[ConflictEdge],
) -> str:
    """Serialize scored claims into a prompt-ready string with conflict annotations.

    Args:
        scored_claims: Claims sorted by final_score descending.
        conflicts: CONTRADICTS / SUPERSEDES edges.

    Returns:
        Formatted string.
    """
    conflict_index: dict[str, list[str]] = {}
    for e in conflicts:
        conflict_index.setdefault(e.source_claim_id, []).append(
            f"{e.relation.value} {e.target_claim_id}: {e.reasoning}"
        )
        conflict_index.setdefault(e.target_claim_id, []).append(
            f"{e.relation.value} by {e.source_claim_id}: {e.reasoning}"
        )

    parts: list[str] = []
    for i, sc in enumerate(scored_claims, 1):
        block = (
            f"[{i}] Claim ID: {sc.claim_id}\n"
            f"    Text: {sc.text}\n"
            f"    Type: {sc.claim_type.value}  |  Source: {sc.source_doc_id}\n"
            f"    Scores: temporal={sc.temporal_score:.2f}  "
            f"authority={sc.authority_score:.2f}  "
            f"graph={sc.graph_evidence_score:.2f}  "
            f"FINAL={sc.final_score:.2f}"
        )
        if sc.temporal_marker:
            block += f"\n    Temporal marker: {sc.temporal_marker}"

        annotations = conflict_index.get(sc.claim_id, [])
        if annotations:
            block += "\n    ** CONFLICTS:"
            for ann in annotations:
                block += f"\n       - {ann}"

        parts.append(block)

    return "\n\n".join(parts)


def build_conflict_summary(
    conflicts: list[ConflictEdge],
    scored_claims: list[ScoredClaim],
) -> str:
    """Build a human-readable conflict summary.

    Args:
        conflicts: CONTRADICTS / SUPERSEDES edges.
        scored_claims: For looking up claim text.

    Returns:
        Summary string, or empty string if no conflicts.
    """
    if not conflicts:
        return ""

    claim_map = {sc.claim_id: sc for sc in scored_claims}
    lines: list[str] = []
    for e in conflicts:
        src_text = _truncate(claim_map.get(e.source_claim_id, None))
        tgt_text = _truncate(claim_map.get(e.target_claim_id, None))
        lines.append(
            f"- {e.source_claim_id} ({src_text}) {e.relation.value} "
            f"{e.target_claim_id} ({tgt_text}): {e.reasoning}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_synthesis_response(raw: str, query: str) -> DeliberationResult | None:
    """Parse raw LLM JSON into a DeliberationResult.

    Args:
        raw: Raw LLM output.
        query: Original query to embed in the result.

    Returns:
        Parsed DeliberationResult, or None on failure.
    """
    try:
        data = _extract_json_object(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("synthesis_parse_failed", raw_head=raw[:200])
        return None

    # answer
    answer = str(data.get("answer", "")).strip()
    if not answer:
        return None

    # confidence level
    conf_raw = str(data.get("confidence", "moderate")).lower()
    if conf_raw not in _VALID_CONFIDENCE:
        conf_raw = "moderate"

    # confidence score
    conf_score = data.get("confidence_score", 0.5)
    try:
        conf_score = max(0.0, min(1.0, float(conf_score)))
    except (TypeError, ValueError):
        conf_score = 0.5

    # reasoning trace
    trace_raw = data.get("reasoning_trace", [])
    if isinstance(trace_raw, list):
        trace = [str(s).strip() for s in trace_raw if str(s).strip()]
    else:
        trace = [str(trace_raw).strip()] if str(trace_raw).strip() else []

    # source attribution
    sa_raw = data.get("source_attribution", [])
    attributions: list[SourceAttribution] = []
    if isinstance(sa_raw, list):
        for item in sa_raw:
            if isinstance(item, dict) and "claim_id" in item:
                rel = item.get("relevance", 0.5)
                try:
                    rel = max(0.0, min(1.0, float(rel)))
                except (TypeError, ValueError):
                    rel = 0.5
                attributions.append(
                    SourceAttribution(
                        claim_id=str(item["claim_id"]),
                        source_doc_id=str(item.get("source_doc_id", "unknown")),
                        relevance=rel,
                    )
                )

    # conflict summary
    conflict_summary = str(data.get("conflict_summary", "")).strip()

    return DeliberationResult(
        query=query,
        answer=answer,
        confidence=ConfidenceLevel(conf_raw),
        confidence_score=conf_score,
        reasoning_trace=trace,
        source_attribution=attributions,
        conflict_summary=conflict_summary,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_conflicts(graph: nx.DiGraph) -> list[ConflictEdge]:
    """Pull CONTRADICTS and SUPERSEDES edges from the graph."""
    result: list[ConflictEdge] = []
    for _, _, data in graph.edges(data=True):
        edge: ConflictEdge = data.get("edge")
        if edge and edge.relation in (RelationType.CONTRADICTS, RelationType.SUPERSEDES):
            result.append(edge)
    return result


def _truncate(claim: ScoredClaim | None, max_len: int = 60) -> str:
    if claim is None:
        return "?"
    text = claim.text
    return text[:max_len] + "..." if len(text) > max_len else text


def _fallback_result(
    query: str, scored_claims: list[ScoredClaim],
) -> DeliberationResult:
    """Build a minimal low-confidence result when LLM parsing fails."""
    top_text = scored_claims[0].text if scored_claims else "Unable to determine answer."
    return DeliberationResult(
        query=query,
        answer=top_text,
        confidence=ConfidenceLevel.LOW,
        confidence_score=0.2,
        reasoning_trace=["LLM synthesis output could not be parsed; using top-scored claim."],
        source_attribution=(
            [SourceAttribution(
                claim_id=scored_claims[0].claim_id,
                source_doc_id=scored_claims[0].source_doc_id,
                relevance=0.5,
            )]
            if scored_claims
            else []
        ),
        conflict_summary="",
    )


def _extract_json_object(text: str) -> dict:
    """Extract the first JSON object from text, handling markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])

    raise ValueError("Unbalanced braces in LLM output")
