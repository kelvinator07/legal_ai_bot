"""
Web search helper for court cases and legal commentary (used by MCP tool).
"""

from __future__ import annotations


def search_legal_cases_online(query: str, max_results: int = 5) -> dict:
    """Search the public web for case law, judgments, and legal news.

    Args:
        query: Search query (e.g. 'Nigeria Supreme Court unfair dismissal').
        max_results: Max number of hits to return (default 5).

    Returns:
        Dict with 'results' list of {title, url, snippet} and optional 'error'.
    """
    q = (query or "").strip()
    if not q:
        return {"error": "Empty query.", "results": []}

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "error": "duckduckgo-search is not installed.",
            "results": [],
        }

    enriched = f"{q} court case judgment legal"
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(enriched, max_results=max(1, min(max_results, 10))))
    except Exception as exc:
        return {"error": str(exc), "results": []}

    results = [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]
    return {
        "query": q,
        "count": len(results),
        "results": results,
        "note": "Web results are third-party summaries, not verified case reports. Verify citations.",
    }
