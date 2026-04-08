"""Shared utilities for legal tool implementations."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "rag" / "knowledge-base"


def load_knowledge_base_file(filename: str) -> str | None:
    """Load a knowledge-base file and return its text, or None if missing."""
    filepath = KNOWLEDGE_BASE_DIR / filename
    try:
        text = filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("[TOOLS] Knowledge base file not found: %s", filepath)
        return None
    logger.debug("[TOOLS] Loaded knowledge base file: %s", filename)
    return text
