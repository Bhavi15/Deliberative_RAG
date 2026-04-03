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
            source_doc_id=source_doc_id,
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

        claims: list[Claim] = []
        for item in items:
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
