"""PDF text extraction + section splitting.

Deliberately simple - see the implementation plan, section 0: GROBID is a
post-hackathon stretch. This header-regex heuristic is the whole of the
MVP's section-detection logic.
"""
from __future__ import annotations

import re

import pymupdf as fitz

_SECTION_HEADERS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "methodology",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
]

_HEADER_RE = re.compile(
    r"^\s*(?:\d+[\.\)]?\s*)?(" + "|".join(re.escape(h) for h in _SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE,
)


def extract_full_text(pdf_path: str) -> str:
    """Return the concatenated text of every page."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def split_into_sections(full_text: str) -> dict[str, str]:
    """Split text into named sections using a header-line heuristic.

    Text before the first recognized header is filed under "preamble".
    A document with no recognizable headers just comes back as
    {"preamble": full_text}.
    """
    sections: dict[str, list[str]] = {"preamble": []}
    current = "preamble"

    for line in full_text.splitlines():
        match = _HEADER_RE.match(line)
        if match:
            current = match.group(1).strip().lower()
            sections.setdefault(current, [])
            continue
        sections[current].append(line)

    return {
        name: "\n".join(body).strip()
        for name, body in sections.items()
        if "\n".join(body).strip()
    }


def get_sections(pdf_path: str) -> dict[str, str]:
    """Convenience: extract text and split it into sections in one call."""
    return split_into_sections(extract_full_text(pdf_path))
