"""Stage 3 — Claim Extraction pipeline node.

Calls the LLM to break each retrieved passage into atomic, self-contained
factual claims.
"""

from src.models import Claim, Passage
from src.pipeline.graph import PipelineState


def claim_extraction_node(state: PipelineState) -> PipelineState:
    """LangGraph node: extract atomic claims from all retrieved passages.

    Args:
        state: Current pipeline state with populated passages.

    Returns:
        Updated state with ``claims`` populated.
    """
    pass


def extract_claims_from_passage(passage: Passage) -> list[Claim]:
    """Call the LLM to extract atomic claims from a single passage.

    Args:
        passage: The passage to extract claims from.

    Returns:
        List of Claim objects grounded in the passage.
    """
    pass


def parse_claims(llm_output: str, passage: Passage) -> list[Claim]:
    """Parse LLM output into a list of Claim objects.

    Args:
        llm_output: Raw text returned by the LLM.
        passage: Source passage for provenance tracking.

    Returns:
        List of Claim instances.
    """
    pass
