"""
Tool: search_legal_database
============================
Looks up statutes and legal provisions from the knowledge base files
in rag/knowledge-base/.

The knowledge base contains real Nigerian legal acts:
    - Labour Act (1974 No. 21)
    - Lagos Tenancy Law (2011)
    - Federal Competition and Consumer Protection Act (2018)
    - Consumer Act
    - Food and Drugs Act
    - Nigeria Constitution (1999)
    - Tenancy Disputes (academic paper)

Presentation example:
    User:  "My boss deducted ₦10,000 from my salary for being late."
    Agent: calls search_legal_database(topic="labor", keyword="deduction")
    Tool searches the Labour Act and returns Section 5 text.
"""

from __future__ import annotations

import logging

from tools.utils import KNOWLEDGE_BASE_DIR, load_knowledge_base_file

logger = logging.getLogger(__name__)

TOPIC_FILE_MAP: dict[str, list[dict[str, str]]] = {
    "labor": [
        {"file": "Labour Act.md", "act_name": "Labour Act, 1974 (No. 21)"},
    ],
    "tenancy": [
        {"file": "Tenancy Law.md", "act_name": "Lagos Tenancy Law, 2011"},
        {"file": "Tenancy Disputes.md", "act_name": "Tenancy Disputes (Research)"},
    ],
    "consumer": [
        {"file": "Federal Consumer Act.md", "act_name": "Federal Competition and Consumer Protection Act, 2018"},
        {"file": "Consumer Act.md", "act_name": "Consumer Act"},
    ],
    "constitution": [
        {"file": "Nigeria Constitution 1999.md", "act_name": "Constitution of the Federal Republic of Nigeria, 1999"},
    ],
    "food": [
        {"file": "Food And Drugs Act.md", "act_name": "Food and Drugs Act"},
    ],
}

ALL_TOPICS = set(TOPIC_FILE_MAP.keys())


def _search_sections(text: str, keyword: str, context_lines: int = 15) -> list[dict]:
    """Search through a legal document for sections matching a keyword.

    Returns matching sections with surrounding context.
    """
    lines = text.split("\n")
    keyword_lower = keyword.lower()
    results = []
    current_section = ""
    current_section_line = 0

    for i, line in enumerate(lines):
        # Track current section heading
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            current_section = stripped.lstrip("#").strip()
            current_section_line = i

        # Check for keyword match
        if keyword_lower in line.lower():
            # Gather context: from section heading to context_lines after match
            start = max(0, current_section_line if current_section_line > i - 20 else i - 5)
            end = min(len(lines), i + context_lines)
            excerpt = "\n".join(lines[start:end]).strip()

            results.append({
                "section": current_section or f"Line {i + 1}",
                "excerpt": excerpt,
                "line_number": i + 1,
            })

    # Deduplicate overlapping results by section
    seen_sections = set()
    unique_results = []
    for r in results:
        if r["section"] not in seen_sections:
            seen_sections.add(r["section"])
            unique_results.append(r)

    return unique_results


def search_legal_database(topic: str, keyword: str = "", section: str = "") -> dict:
    """Search Nigerian legal statutes from the knowledge base by topic and keyword.

    Use this tool when the user asks about their legal rights, whether
    something is legal/illegal, or what the law says about a topic.

    Args:
        topic:   Legal topic — one of: labor, tenancy, consumer, constitution, food.
        keyword: Keyword to search for within the act (e.g. "deduction", "eviction", "notice").
        section: Optional specific section number to look up (e.g. "5", "13").

    Returns:
        A dict with matching statute sections and their text.
    """
    topic_lower = topic.lower()

    if topic_lower not in TOPIC_FILE_MAP:
        return {
            "found": False,
            "message": f"No data for topic '{topic}'. Available topics: {', '.join(sorted(ALL_TOPICS))}.",
        }

    if not keyword and not section:
        return {
            "found": False,
            "message": "Please provide a keyword or section number to search for.",
        }

    all_results = []

    for source in TOPIC_FILE_MAP[topic_lower]:
        text = load_knowledge_base_file(source["file"])
        if text is None:
            continue

        search_term = section if section else keyword

        matches = _search_sections(text, search_term)
        for match in matches:
            match["source_act"] = source["act_name"]
            match["source_file"] = source["file"]

        all_results.extend(matches)

    if not all_results:
        search_desc = f"section '{section}'" if section else f"keyword '{keyword}'"
        return {
            "found": False,
            "message": f"No results matching {search_desc} under topic '{topic}'.",
        }

    # Limit to top 5 most relevant results
    return {
        "found": True,
        "topic": topic,
        "search_term": section or keyword,
        "results": all_results[:5],
        "count": len(all_results[:5]),
        "note": "Excerpts from actual Nigerian legal acts in rag/knowledge-base/.",
    }
