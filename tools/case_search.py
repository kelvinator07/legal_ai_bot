"""
Tool: find_similar_cases
=========================
Searches the knowledge base files in rag/knowledge-base/ for sections
relevant to the user's legal situation.

Instead of a hardcoded case database, this tool searches across the actual
Nigerian legal acts to find provisions that apply to the user's scenario.

In production, this would use embeddings + vector search (via the rag/ module)
to find semantically similar content across documents.

Presentation example:
    User:  "I was fired without warning. What can I expect?"
    Agent: calls find_similar_cases(description="fired without warning")
    Tool searches Labour Act for termination provisions → returns Section 11.
"""

from __future__ import annotations

import logging

from tools.utils import load_knowledge_base_file

logger = logging.getLogger(__name__)

# Map common legal scenarios to relevant files and search terms
SCENARIO_KEYWORDS: dict[str, list[dict]] = {
    "fired": [
        {"file": "Labour Act.md", "terms": ["terminat", "notice", "dismiss", "contract"]},
    ],
    "dismissed": [
        {"file": "Labour Act.md", "terms": ["terminat", "notice", "dismiss"]},
    ],
    "salary": [
        {"file": "Labour Act.md", "terms": ["wages", "payment", "deduction", "salary"]},
    ],
    "deduction": [
        {"file": "Labour Act.md", "terms": ["deduction", "wages"]},
    ],
    "eviction": [
        {"file": "Tenancy Law.md", "terms": ["possession", "evict", "notice", "quit"]},
    ],
    "landlord": [
        {"file": "Tenancy Law.md", "terms": ["landlord", "tenant", "rent", "possession"]},
    ],
    "rent": [
        {"file": "Tenancy Law.md", "terms": ["rent", "tenancy", "notice", "arrears"]},
    ],
    "refund": [
        {"file": "Federal Consumer Act.md", "terms": ["refund", "return", "defective"]},
        {"file": "Consumer Act.md", "terms": ["refund", "return", "defective"]},
    ],
    "defective": [
        {"file": "Federal Consumer Act.md", "terms": ["defective", "return", "refund", "liability"]},
    ],
    "consumer": [
        {"file": "Federal Consumer Act.md", "terms": ["consumer", "right", "protection"]},
    ],
    "rights": [
        {"file": "Nigeria Constitution 1999.md", "terms": ["right", "freedom", "liberty"]},
    ],
    "arrest": [
        {"file": "Nigeria Constitution 1999.md", "terms": ["arrest", "detention", "liberty"]},
    ],
    "overtime": [
        {"file": "Labour Act.md", "terms": ["overtime", "hours", "work"]},
    ],
    "leave": [
        {"file": "Labour Act.md", "terms": ["holiday", "leave", "annual", "sick"]},
    ],
    "contract": [
        {"file": "Labour Act.md", "terms": ["contract", "employment", "terms"]},
        {"file": "Federal Consumer Act.md", "terms": ["contract", "agreement", "terms"]},
    ],
    "food": [
        {"file": "Food And Drugs Act.md", "terms": ["food", "drug", "manufacture", "sale"]},
    ],
}


def _find_relevant_sections(text: str, terms: list[str], max_sections: int = 3) -> list[dict]:
    """Find sections in a legal document that match the given terms."""
    lines = text.split("\n")
    sections: list[dict] = []
    current_section = ""
    current_section_start = 0
    current_section_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Detect section headings
        if stripped.startswith("## ") or stripped.startswith("### "):
            # Save previous section if it had matches
            if current_section and current_section_lines:
                section_text = "\n".join(current_section_lines)
                score = sum(
                    section_text.lower().count(term.lower())
                    for term in terms
                )
                if score > 0:
                    sections.append({
                        "section": current_section,
                        "excerpt": section_text[:800].strip(),
                        "relevance_score": score,
                    })

            current_section = stripped.lstrip("#").strip()
            current_section_start = i
            current_section_lines = []
        else:
            current_section_lines.append(line)

    # Don't forget the last section
    if current_section and current_section_lines:
        section_text = "\n".join(current_section_lines)
        score = sum(section_text.lower().count(term.lower()) for term in terms)
        if score > 0:
            sections.append({
                "section": current_section,
                "excerpt": section_text[:800].strip(),
                "relevance_score": score,
            })

    # Sort by relevance and return top results
    sections.sort(key=lambda s: s["relevance_score"], reverse=True)
    return sections[:max_sections]


def find_similar_cases(
    description: str,
    topic: str = "",
    max_results: int = 5,
) -> dict:
    """Find relevant legal provisions for the user's situation from the knowledge base.

    Use this tool when the user describes a dispute and wants to know
    what the law says about their situation, what provisions apply, or
    what outcomes to expect.

    Args:
        description: Plain-language description of the user's situation.
        topic:       Optional — filter by topic (e.g. "labor", "tenancy", "consumer").
        max_results: Maximum number of relevant sections to return (default 5).

    Returns:
        A dict with relevant legal provisions from the knowledge base.
    """
    if not description or not description.strip():
        return {"error": "Please describe your legal situation so I can find relevant provisions."}

    desc_lower = description.lower()

    # Determine which files and terms to search
    files_to_search: list[dict] = []

    # Match scenario keywords from description
    for keyword, sources in SCENARIO_KEYWORDS.items():
        if keyword in desc_lower:
            for source in sources:
                if source not in files_to_search:
                    files_to_search.append(source)

    # If no specific match, search all files with words from description
    if not files_to_search:
        desc_words = [w for w in desc_lower.split() if len(w) > 3]
        all_files = [
            "Labour Act.md",
            "Tenancy Law.md",
            "Federal Consumer Act.md",
            "Consumer Act.md",
            "Nigeria Constitution 1999.md",
            "Food And Drugs Act.md",
        ]
        files_to_search = [{"file": f, "terms": desc_words} for f in all_files]

    # Search each file
    all_results = []
    for source in files_to_search:
        text = load_knowledge_base_file(source["file"])
        if text is None:
            continue

        sections = _find_relevant_sections(text, source["terms"], max_sections=3)
        for section in sections:
            section["source_file"] = source["file"]
            section["source_act"] = text.split("\n")[0].strip("# ").strip()

        all_results.extend(sections)

    # Sort all results by relevance and limit
    all_results.sort(key=lambda s: s["relevance_score"], reverse=True)
    top_results = all_results[:max_results]

    if not top_results:
        return {
            "found": False,
            "message": "No relevant provisions found. Try describing your situation with different terms.",
        }

    # Clean up results for output
    for r in top_results:
        del r["relevance_score"]

    return {
        "found": True,
        "query": description,
        "provisions": top_results,
        "count": len(top_results),
        "note": "Provisions sourced from actual Nigerian legal acts in rag/knowledge-base/.",
    }
