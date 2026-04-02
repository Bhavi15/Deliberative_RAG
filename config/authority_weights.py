"""Source authority weight configuration.

Maps source domain patterns to pre-assigned credibility scores in [0, 1].
Higher scores indicate more authoritative sources.
"""

# Domain-level authority scores
DOMAIN_WEIGHTS: dict[str, float] = {
    # Academic / peer-reviewed
    "arxiv.org": 0.85,
    "pubmed.ncbi.nlm.nih.gov": 0.90,
    "scholar.google.com": 0.80,
    # News / journalism
    "reuters.com": 0.80,
    "apnews.com": 0.80,
    "bbc.com": 0.75,
    "nytimes.com": 0.72,
    # Government / official
    "gov": 0.88,
    "who.int": 0.90,
    "cdc.gov": 0.90,
    # Finance
    "sec.gov": 0.92,
    "bloomberg.com": 0.78,
    "ft.com": 0.78,
    # Default fallback for unknown sources
    "default": 0.50,
}

# Recency decay half-life in days (used by temporal scoring)
RECENCY_HALF_LIFE_DAYS: int = 365


def get_authority_score(source_url: str) -> float:
    """Return the authority score for a given source URL.

    Matches the URL against known domain patterns and falls back to the
    default weight when no match is found.

    Args:
        source_url: Canonical URL or domain of the source document.

    Returns:
        Authority score in [0, 1].
    """
    pass
