"""Searcher agent — decomposes query, searches arXiv + Semantic Scholar,
deduplicates, tags with subtopics, and returns new PaperRecords."""

from dotenv import load_dotenv
load_dotenv()

import json

from pydantic import BaseModel
from rapidfuzz import fuzz

from state.schemas import CycleState, PaperRecord
from tools.arxiv_client import search_arxiv
from tools.llm import LLMError, generate_json
from tools.semantic_scholar_client import search_semantic_scholar


class _SubtopicList(BaseModel):
    subtopics: list[str]


class _PaperTags(BaseModel):
    tags_by_index: dict[str, list[str]]


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


def tag_papers(papers: list[PaperRecord], subtopics: list[str]) -> list[PaperRecord]:
    """One cheap LLM call per cycle: tags each paper against the subtopics.

    Tagging failure shouldn't sink the whole cycle - on LLMError, papers
    just come back untagged rather than crashing the Searcher.
    """
    if not papers:
        return papers

    paper_list_text = "\n".join(f"{i}. {p.title}" for i, p in enumerate(papers))
    prompt = (
        f"Subtopics: {json.dumps(subtopics)}\n\nPapers:\n{paper_list_text}\n\n"
        "For each paper (by its number above, as a string key), list which "
        "subtopics it relates to."
    )

    try:
        result = generate_json(prompt, _PaperTags)
    except LLMError as exc:
        print(f"[searcher] tagging failed, leaving papers untagged: {exc}")
        return papers

    for i, paper in enumerate(papers):
        paper.subtopic_tags = result.tags_by_index.get(str(i), [])

    return papers


def search_all_terms(terms: list[str], max_results_per_source: int = 3) -> list[PaperRecord]:
    """Searches both arXiv and Semantic Scholar for every term given."""
    all_papers = []
    for term in terms:
        all_papers.extend(search_arxiv(term, max_results=max_results_per_source))
        all_papers.extend(search_semantic_scholar(term, max_results=max_results_per_source))
    return all_papers


def searcher_node(state: CycleState, refiner_decision: dict = None) -> dict:
    """
    Main Searcher entry point.
    state: the real CycleState pydantic object.
    refiner_decision: optional dict from Strategy Refiner (Cycle 2+), shaped as:
        {"decision": "continue"|"stop", "new_query_terms": [...], "papers_to_reexamine": [...], "rationale": "..."}
    """
    if refiner_decision:
        query_terms = refiner_decision["new_query_terms"]
    else:
        query_terms = state.subtopics  # Cycle 1 fallback

    existing_paper_ids = {p.id for p in state.papers}

    combined = search_all_terms(query_terms)
    deduped = deduplicate_papers(combined)
    tagged = tag_papers(deduped, state.subtopics)  # always tag against ORIGINAL subtopics
    new_papers = [p for p in tagged if p.id not in existing_paper_ids]

    return {"new_papers": new_papers}