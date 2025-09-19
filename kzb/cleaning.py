"""Utilities for cleaning Kol-Zchut pages before translation and embedding."""
from __future__ import annotations

import re
from typing import Iterable

FOOTER_PATTERNS = (
    r"^\s*קישורים\s+חיצוניים",
    r"^\s*ראו\s+גם",
    r"^\s*\.?\s*©",
    r"כל\s+הזכויות\s+שמורות",
)

URL_PATTERN = re.compile(r"https?://\S+")
MULTISPACE_PATTERN = re.compile(r"[ \t]+")


def _drop_footer_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(re.search(pattern, stripped) for pattern in FOOTER_PATTERNS):
            break
        cleaned.append(line)
    return cleaned


def clean_page_text(text: str) -> str:
    """Return a cleaned version of the raw page text."""
    text = URL_PATTERN.sub("", text)
    text = text.replace("\u200f", "")
    lines = text.splitlines()
    lines = [line for line in lines if line.strip()]
    lines = _drop_footer_lines(lines)
    text = "\n".join(lines)
    text = MULTISPACE_PATTERN.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
