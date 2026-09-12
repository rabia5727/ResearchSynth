"""Agent 2 - PDF Reader.

Input:  a PaperRecord (+ optionally a local pdf_path, if already downloaded)
Output: ExtractedFinding[] - structured findings, never raw text, so
        downstream agents (Contradiction Detector, Synthesis Writer) stay
        fast and every claim is traceable to a section of a specific paper.

Per paper: download (skip + mark pdf_accessible=False on failure) -> extract
text -> split into sections -> one LLM call -> ExtractedFinding[]. Results
are cached by paper_id so a paper already processed in an earlier research
cycle is never re-read or re-billed.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from pydantic import BaseModel

from state.schemas import ExtractedFinding, PaperRecord
from tools.llm import LLMError, generate_json
from tools.pdf_parser import get_sections

_SYSTEM_PROMPT = (
    "You are the PDF Reader agent in a literature-review pipeline. Extract "
    "only what is explicitly stated in the given paper sections. Never "
    "invent a claim, method, dataset, metric, or value that is not present "
    "in the text. If a field genuinely is not stated, omit it."
)


class _FindingDraft(BaseModel):
    """What we ask the LLM for - paper_id/id get added afterwards, since the
    model shouldn't be trusted to invent identifiers."""

    claim: str
    method: str | None = None
    dataset: str | None = None
    metric: str | None = None
    value: str | None = None
    limitations: str | None = None
    section_source: str


class _FindingsExtraction(BaseModel):
    findings: list[_FindingDraft]


def fetch_pdf(paper: PaperRecord, *, dest_dir: str | Path = ".cache/pdfs", timeout: int = 20) -> str | None:
    """Download paper.url to a local file. Returns the local path, or None on
    failure (paywalled / 403 / 404 / network error) - the caller marks
    pdf_accessible=False and moves on rather than failing the whole cycle.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{_safe_name(paper.id)}.pdf"
    if dest.exists():
        return str(dest)

    try:
        response = requests.get(paper.url, timeout=timeout, headers={"User-Agent": "ResearchSynth/0.1"})
        response.raise_for_status()
        dest.write_bytes(response.content)
        return str(dest)
    except requests.RequestException as exc:
        print(f"[pdf_reader] could not fetch {paper.id}: {exc}")
        return None


def extract_findings(
    paper: PaperRecord,
    pdf_path: str | None = None,
    *,
    cache_dir: str | Path = ".cache/findings",
) -> list[ExtractedFinding]:
    """The agent's main entry point. Returns [] (not an exception) when the
    paper is inaccessible or extraction fails - callers should keep going
    with the rest of the batch, per the plan's paywall-handling risk mitigation.
    """
    cache_dir = Path(cache_dir)
    cached = _load_from_cache(paper.id, cache_dir)
    if cached is not None:
        return cached

    if pdf_path is None:
        pdf_path = fetch_pdf(paper)
        if pdf_path is None:
            paper.pdf_accessible = False
            return []

    sections = get_sections(pdf_path)
    if not sections:
        print(f"[pdf_reader] no extractable text/sections for {paper.id}")
        return []

    prompt = _build_prompt(paper, sections)
    try:
        extraction = generate_json(prompt, _FindingsExtraction, system=_SYSTEM_PROMPT)
    except LLMError as exc:
        print(f"[pdf_reader] extraction failed for {paper.id}: {exc}")
        return []

    findings = [
        ExtractedFinding(
            id=f"{paper.id}-f{i + 1}",
            paper_id=paper.id,
            claim=draft.claim,
            method=draft.method,
            dataset=draft.dataset,
            metric=draft.metric,
            value=draft.value,
            limitations=draft.limitations,
            section_source=draft.section_source,
        )
        for i, draft in enumerate(extraction.findings)
    ]
    _save_to_cache(paper.id, findings, cache_dir)
    return findings


def _build_prompt(paper: PaperRecord, sections: dict[str, str]) -> str:
    section_text = "\n\n".join(f"## {name.upper()}\n{body}" for name, body in sections.items())
    return (
        f"Paper title: {paper.title}\n\n"
        f"{section_text}\n\n"
        "Extract every distinct finding as a structured object: claim, "
        "method, dataset, metric, value, limitations, and which section it "
        "came from (section_source). A paper may report more than one finding."
    )


def _safe_name(paper_id: str) -> str:
    return paper_id.replace("/", "_").replace(":", "_")


def _cache_path(paper_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_safe_name(paper_id)}.json"


def _load_from_cache(paper_id: str, cache_dir: Path) -> list[ExtractedFinding] | None:
    path = _cache_path(paper_id, cache_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ExtractedFinding.model_validate(item) for item in data]


def _save_to_cache(paper_id: str, findings: list[ExtractedFinding], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(paper_id, cache_dir).write_text(
        json.dumps([f.model_dump() for f in findings], indent=2), encoding="utf-8"
    )
