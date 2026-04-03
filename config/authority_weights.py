"""Source authority weight configuration.

Maps source type labels to pre-assigned credibility scores in [0, 1].
Higher scores indicate more authoritative sources.
"""

# Source-type authority scores — keyed by the label stored in document metadata.
AUTHORITY_WEIGHTS: dict[str, float] = {
    "official": 1.0,            # Central banks, government filings (SEC, FOMC, BIS)
    "peer_reviewed": 0.9,       # Academic journals
    "institutional": 0.8,       # Major bank research (Goldman, JPM)
    "encyclopedic": 0.75,       # Wikipedia, established references
    "news_major": 0.6,          # Reuters, Bloomberg
    "news_minor": 0.4,
    "opinion": 0.3,
    "blog": 0.15,
    "counter_evidence": 0.5,    # Synthetic contradictions from our dataset
    "generated_counter": 0.5,
}

# Default score when the source type is unknown
DEFAULT_AUTHORITY_WEIGHT: float = 0.3

# Recency decay half-life in days (used by temporal scoring)
RECENCY_HALF_LIFE_DAYS: int = 365
