"""Stage 3 — Claim Extraction.

Calls the LLM to break each retrieved passage into atomic, self-contained
factual claims and returns validated Claim pydantic models.
"""

from __future__ import annotations

import json
import re
import uuid

import structlog

from src.schemas import Claim, ClaimType
from src.utils.llm import LLMClient
from src.utils.prompts import load_prompt

log = structlog.get_logger()

_VALID_CLAIM_TYPES = {t.value for t in ClaimType}


class ClaimExtractor:
    """Extracts atomic claims from text passages via an LLM."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialise the extractor with an LLM client.

        Args:
            llm: Configured LLMClient instance for calling Claude.
        """
        self._llm = llm

    def extract_claims(self, passage: str, doc_metadata: dict) -> list[Claim]:
        """Extract atomic claims from a single passage.

        Sends the passage to the LLM with the claim_extraction prompt,
        parses the JSON response into Claim models.  On malformed JSON the
        LLM is retried once; if the retry also fails an empty list is returned.

        Args:
            passage: Raw passage text.
            doc_metadata: Dict that must contain ``"source_doc_id"``.  Any
                additional keys are ignored.

        Returns:
            List of validated Claim objects.
        """
        source_doc_id = doc_metadata.get("source_doc_id", "unknown")
        prompt = load_prompt(
            "claim_extraction",
            passage=passage,
            query="",
        )

        raw = self._llm.invoke(prompt)
        claims = self._parse_response(raw, source_doc_id)

        if claims is not None:
            return claims

        # Retry once on malformed JSON
        log.warning("claim_extraction_retry", source_doc_id=source_doc_id)
        raw_retry = self._llm.invoke(prompt)
        claims_retry = self._parse_response(raw_retry, source_doc_id)

        if claims_retry is not None:
            return claims_retry

        log.error("claim_extraction_failed", source_doc_id=source_doc_id)
        return []

    def extract_claims_batch(
        self,
        passages: list[dict],
    ) -> list[list[Claim]]:
        """Extract claims from multiple passages.

        Args:
            passages: List of dicts each containing ``"text"`` and
                ``"source_doc_id"`` keys.

        Returns:
            List of claim lists, one per input passage.
        """
        results: list[list[Claim]] = []
        for p in passages:
            text = p.get("text", "")
            if not text.strip():
                results.append([])
                continue
            claims = self.extract_claims(text, p)
            results.append(claims)
        return results

    def extract_claims_combined(
        self, passages: list[dict], query: str = "",
    ) -> list[Claim]:
        """Extract claims from ALL passages in a single LLM call.

        Combines all passage texts into one prompt to minimise LLM round-trips
        (critical for slow local inference).

        Args:
            passages: List of dicts with ``"text"`` and ``"source_doc_id"``.
            query: The original user query for relevance filtering.

        Returns:
            Flat list of Claims from all passages.
        """
        if not passages:
            return []

        # Build combined passage block with source IDs in headers
        source_lookup: dict[str, str] = {}
        combined = ""
        for i, p in enumerate(passages):
            text = p.get("text", "").strip()
            source = p.get("source_doc_id", f"doc_{i}")
            source_lookup[source] = source
            if text:
                combined += f"\n--- Passage {i+1} (source_id: {source}) ---\n{text}\n"

        prompt = load_prompt(
            "claim_extraction",
            passage=combined,
            query=query,
        )

        raw = self._llm.invoke(prompt)
        claims = self._parse_response_combined(raw, source_lookup)

        if claims is not None:
            return claims

        # Retry once
        log.warning("claim_extraction_combined_retry")
        raw_retry = self._llm.invoke(prompt)
        claims_retry = self._parse_response_combined(raw_retry, source_lookup)

        if claims_retry is not None:
            return claims_retry

        log.error("claim_extraction_combined_failed")
        return []

    def _parse_response_combined(
        self, raw: str, source_lookup: dict[str, str],
    ) -> list[Claim] | None:
        """Parse combined extraction response, mapping source_id per claim.

        Args:
            raw: Raw LLM output (JSON array).
            source_lookup: Valid source IDs from passages.

        Returns:
            List of Claims with correct source_doc_id, or None on failure.
        """
        try:
            items = _extract_json_array(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("json_parse_failed", raw_head=raw[:200])
            return None

        valid_sources = set(source_lookup.keys())
        fallback_source = next(iter(valid_sources), "unknown")

        max_claims = 5
        claims: list[Claim] = []
        for item in items[:max_claims]:
            if not isinstance(item, dict) or "text" not in item:
                continue

            # Map source_id from LLM response to actual source doc ID
            raw_source = str(item.get("source_id", "")).strip()
            if raw_source in valid_sources:
                source_doc_id = raw_source
            else:
                source_doc_id = fallback_source

            claim_type_raw = item.get("claim_type", "fact")
            if claim_type_raw not in _VALID_CLAIM_TYPES:
                claim_type_raw = "fact"

            temporal = item.get("temporal_marker")
            if temporal is not None:
                temporal = str(temporal).strip() or None

            confidence = item.get("confidence_in_extraction", 0.8)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 0.8

            claims.append(
                Claim(
                    claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                    text=item["text"].strip(),
                    claim_type=ClaimType(claim_type_raw),
                    source_doc_id=source_doc_id,
                    temporal_marker=temporal,
                    confidence_in_extraction=confidence,
                )
            )

        return claims

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, source_doc_id: str) -> list[Claim] | None:
        """Try to parse raw LLM output into a list of Claims.

        Args:
            raw: Raw text from the LLM (should be a JSON array).
            source_doc_id: Fallback document ID for provenance.

        Returns:
            List of Claims on success, ``None`` on parse failure.
        """
        try:
            items = _extract_json_array(raw)
        except (json.JSONDecodeError, ValueError):
            log.warning("json_parse_failed", raw_head=raw[:200])
            return None

        max_claims = 5  # Hard cap to limit downstream LLM calls
        claims: list[Claim] = []
        for item in items[:max_claims]:
            if not isinstance(item, dict) or "text" not in item:
                continue

            claim_type_raw = item.get("claim_type", "fact")
            if claim_type_raw not in _VALID_CLAIM_TYPES:
                claim_type_raw = "fact"

            temporal = item.get("temporal_marker")
            if temporal is not None:
                temporal = str(temporal).strip() or None

            confidence = item.get("confidence_in_extraction", 0.8)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 0.8

            claims.append(
                Claim(
                    claim_id=f"claim_{uuid.uuid4().hex[:8]}",
                    text=item["text"].strip(),
                    claim_type=ClaimType(claim_type_raw),
                    source_doc_id=source_doc_id,
                    temporal_marker=temporal,
                    confidence_in_extraction=confidence,
                )
            )

        return claims


def _extract_json_array(text: str) -> list[dict]:
    """Extract and parse the first JSON array found in *text*.

    Handles common LLM quirks: markdown fences, leading prose, trailing
    commentary.

    Args:
        text: Raw LLM output potentially containing a JSON array.

    Returns:
        Parsed list of dicts.

    Raises:
        ValueError: If no JSON array can be found.
        json.JSONDecodeError: If the array is malformed.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")

    # Find the first '[' ... ']' pair
    start = cleaned.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM output")

    depth = 0
    end = start
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        raise ValueError("Unbalanced brackets in LLM output")

    return json.loads(cleaned[start:end])
