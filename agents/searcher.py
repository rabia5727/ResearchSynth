"""Searcher agent — decomposes query, searches arXiv + Semantic Scholar,
deduplicates, tags with subtopics, and returns new PaperRecords."""

from dotenv import load_dotenv
load_dotenv()

import json
import os

from google import genai
from rapidfuzz import fuzz

from state.schemas import CycleState, PaperRecord
from tools.arxiv_client import search_arxiv
from tools.semantic_scholar_client import search_semantic_scholar


# --- Gemini setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.6-flash"


def decompose_query(query: str) -> list[str]:
    """One-time call (Cycle 0): breaks the research question into 3-6 subtopics."""
    prompt = f"""You are helping break a research question into 3 to 6 smaller subtopics
for academic paper searching.

Research question: "{query}"

Return ONLY a JSON list of subtopic strings, nothing else.
Example format: ["subtopic 1", "subtopic 2", "subtopic 3"]
"""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


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
    """One cheap Gemini call per cycle: tags each paper against the subtopics."""
    if not papers:
        return papers

    paper_list_text = "\n".join([f"{i}. {p.title}" for i, p in enumerate(papers)])

    prompt = f"""You are tagging academic papers with relevant subtopics.

Subtopics: {json.dumps(subtopics)}

Papers:
{paper_list_text}

For each paper, return which subtopics (from the list above) it relates to.
Return ONLY a JSON object where each key is the paper's number (as a string)
and the value is a list of matching subtopic strings.

Example format:
{{"0": ["subtopic 1"], "1": ["subtopic 2", "subtopic 3"]}}
"""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    tags_by_index = json.loads(text)

    for i, paper in enumerate(papers):
        paper.subtopic_tags = tags_by_index.get(str(i), [])

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