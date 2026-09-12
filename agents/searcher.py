"""Searcher agent — decomposes query, searches arXiv + Semantic Scholar,
deduplicates, tags with subtopics, and returns new PaperRecords."""

from dotenv import load_dotenv
load_dotenv()

import json
import time
from typing import Callable

from pydantic import BaseModel
from rapidfuzz import fuzz

from state.schemas import CycleState, PaperRecord
from tools.arxiv_client import search_arxiv
from tools.llm import LLMError, generate_json
from tools.semantic_scholar_client import search_semantic_scholar


class _SubtopicList(BaseModel):
    subtopics: list[str]


class _PaperTag(BaseModel):
    index: int
    subtopics: list[str]


class _PaperTags(BaseModel):
    # A list of {index, subtopics} rather than dict[str, list[str]] - Gemini's
    # Developer API rejects response schemas using additionalProperties (the
    # JSON Schema shape a dict-typed field produces), so a free-form mapping
    # isn't usable as a structured-output schema at all.
    tags: list[_PaperTag]


def decompose_query(query: str) -> list[str]:
    """One-time call (Cycle 0): breaks the research question into 3-6 subtopics.

    Goes through the shared generate_json() abstraction (tools/llm.py) so
    this works with LLM_PROVIDER=mock/gemini/groq like every other agent -
    no direct Gemini client, no hand-rolled ```json fence stripping.
    """
    prompt = (
        "Break this research question into 3 to 6 smaller subtopics for "
        f"academic paper searching.\n\nResearch question: \"{query}\""
    )
    result = generate_json(prompt, _SubtopicList)
    return result.subtopics


def deduplicate_papers(papers: list[PaperRecord], title_similarity_threshold: int = 90) -> list[PaperRecord]:
    """Removes duplicate papers by ID match or fuzzy title match."""
    unique_papers = []
    for paper in papers:
        is_duplicate = False
        for existing in unique_papers:
            if paper.id == existing.id:
                is_duplicate = True
                break
            similarity = fuzz.ratio(paper.title.lower(), existing.title.lower())
            if similarity >= title_similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_papers.append(paper)
    return unique_papers


def tag_papers(
    papers: list[PaperRecord],
    subtopics: list[str],
    on_progress: Callable[[str], None] | None = None,
) -> list[PaperRecord]:
    """One cheap LLM call per cycle: tags each paper against the subtopics.

    Tagging failure shouldn't sink the whole cycle - on LLMError, papers
    just come back untagged rather than crashing the Searcher.
    """
    if not papers:
        return papers

    if on_progress:
        on_progress(f"Tagging {len(papers)} paper(s) against subtopics...")

    paper_list_text = "\n".join(f"{i}. {p.title}" for i, p in enumerate(papers))
    prompt = (
        f"Subtopics: {json.dumps(subtopics)}\n\nPapers:\n{paper_list_text}\n\n"
        "For each paper (by its number above), list which subtopics it relates to."
    )

    try:
        result = generate_json(prompt, _PaperTags)
    except LLMError as exc:
        print(f"[searcher] tagging failed, leaving papers untagged: {exc}")
        if on_progress:
            on_progress("Tagging failed - continuing with untagged papers")
        return papers

    tags_by_index = {tag.index: tag.subtopics for tag in result.tags}
    for i, paper in enumerate(papers):
        paper.subtopic_tags = tags_by_index.get(i, [])

    return papers


def search_all_terms(
    terms: list[str],
    max_results_per_source: int = 3,
    on_progress: Callable[[str], None] | None = None,
) -> list[PaperRecord]:
    """Searches both arXiv and Semantic Scholar for every term given.

    A real decomposition can easily produce 5-6 subtopics, each firing two
    searches - a short pause between terms is enough to avoid tripping both
    APIs' rate limits in the first place, rather than relying on retry/backoff
    to recover every time.
    """
    all_papers = []
    for i, term in enumerate(terms):
        if on_progress:
            on_progress(f"Term {i + 1}/{len(terms)}: \"{term}\"")
        if i > 0:
            time.sleep(1)
        all_papers.extend(search_arxiv(term, max_results=max_results_per_source, on_progress=on_progress))
        all_papers.extend(search_semantic_scholar(term, max_results=max_results_per_source, on_progress=on_progress))
    return all_papers


def searcher_node(
    state: CycleState,
    refiner_decision: dict = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """
    Main Searcher entry point.
    state: the real CycleState pydantic object.
    refiner_decision: optional dict from Strategy Refiner (Cycle 2+), shaped as:
        {"decision": "continue"|"stop", "new_query_terms": [...], "papers_to_reexamine": [...], "rationale": "..."}
    on_progress: optional callback invoked with short status strings as the
        search happens - lets a UI show live progress instead of a blank
        wait during the (often the slowest) search phase.
    """
    if refiner_decision:
        query_terms = refiner_decision["new_query_terms"]
    else:
        query_terms = state.subtopics  # Cycle 1 fallback

    existing_paper_ids = {p.id for p in state.papers}

    combined = search_all_terms(query_terms, on_progress=on_progress)
    deduped = deduplicate_papers(combined)
    tagged = tag_papers(deduped, state.subtopics, on_progress=on_progress)  # always tag against ORIGINAL subtopics
    new_papers = [p for p in tagged if p.id not in existing_paper_ids]

    return {"new_papers": new_papers}