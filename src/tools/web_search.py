"""Web search tool using Tavily API.

Provides real-time web search results that the deep research agent
can use alongside the indexed knowledge base.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from tavily import TavilyClient

from config.settings import get_settings

log = structlog.get_logger()


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information on a topic.

    Returns formatted search results with titles, URLs, content snippets,
    and publication dates that can be fed into the conflict detection pipeline.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1-10).

    Returns:
        Formatted string of search results with metadata.
    """
    settings = get_settings()
    api_key = settings.tavily_api_key
    if not api_key:
        return "[Web search unavailable — TAVILY_API_KEY not set]"

    max_results = min(max(1, max_results), 10)

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,
            search_depth="advanced",
        )
    except Exception as exc:
        log.error("web_search_failed", query=query[:60], error=str(exc))
        return f"[Web search error: {exc}]"

    results = response.get("results", [])
    if not results:
        return f"[No web results found for: {query}]"

    formatted_parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")
        published = r.get("published_date", "")

        block = (
            f"[Result {i}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Published: {published or 'unknown'}\n"
            f"Content: {content}\n"
        )
        formatted_parts.append(block)

    log.info("web_search_done", query=query[:60], results=len(results))
    return "\n---\n".join(formatted_parts)


def web_search_structured(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web and return structured results for pipeline integration.

    Unlike :func:`web_search` which returns a formatted string, this returns
    dicts compatible with the retrieval pipeline's document format.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of document dicts with ``text``, ``score``, ``source_type``,
        ``publication_date``, ``url``, and ``source`` fields.
    """
    settings = get_settings()
    api_key = settings.tavily_api_key
    if not api_key:
        return []

    max_results = min(max(1, max_results), 10)

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=False,
            search_depth="advanced",
        )
    except Exception as exc:
        log.error("web_search_structured_failed", query=query[:60], error=str(exc))
        return []

    documents: list[dict[str, Any]] = []
    for r in response.get("results", []):
        url = r.get("url", "")
        domain = _extract_domain(url)

        documents.append({
            "id": f"web_{hash(url) & 0xFFFFFFFF:08x}",
            "text": r.get("content", ""),
            "score": r.get("score", 0.5),
            "source_type": _classify_domain_authority(domain),
            "publication_date": r.get("published_date"),
            "url": url,
            "title": r.get("title", ""),
            "source": "web_search",
            "domain": domain,
        })

    log.info("web_search_structured_done", query=query[:60], results=len(documents))
    return documents


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _classify_domain_authority(domain: str) -> str:
    """Map a web domain to an authority classification.

    Args:
        domain: Lowercase domain name (e.g. ``"reuters.com"``).

    Returns:
        Authority type string matching keys in ``AUTHORITY_WEIGHTS``.
    """
    official_domains = {
        "gov", "edu", "sec.gov", "who.int", "un.org",
        "whitehouse.gov", "congress.gov", "nih.gov", "cdc.gov",
    }
    major_news = {
        "reuters.com", "apnews.com", "bloomberg.com", "wsj.com",
        "nytimes.com", "bbc.com", "bbc.co.uk", "ft.com",
        "economist.com", "theguardian.com", "washingtonpost.com",
    }
    encyclopedic = {"wikipedia.org", "britannica.com", "scholarpedia.org"}
    peer_reviewed = {
        "nature.com", "science.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com", "ieee.org", "acm.org", "springer.com",
    }
    institutional = {
        "imf.org", "worldbank.org", "federalreserve.gov",
        "ecb.europa.eu", "bis.org",
    }

    # Check TLD-based classification
    tld = domain.split(".")[-1] if domain else ""
    if tld in {"gov", "edu", "mil"}:
        return "official"

    # Check exact domain matches
    if domain in official_domains:
        return "official"
    if domain in peer_reviewed:
        return "peer_reviewed"
    if domain in institutional:
        return "institutional"
    if domain in encyclopedic:
        return "encyclopedic"
    if domain in major_news:
        return "news_major"

    # Heuristic: if domain contains "news" or "press"
    if any(kw in domain for kw in ("news", "press", "journal", "times")):
        return "news_minor"
    if any(kw in domain for kw in ("blog", "medium.com", "substack")):
        return "blog"

    return "news_minor"
