"""Semantic Scholar search client — queries the Graph API and returns PaperRecords."""

import time
from typing import Callable

import requests

from state.schemas import PaperRecord
from tools.http_utils import get_with_hard_timeout


def search_semantic_scholar(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
    on_progress: Callable[[str], None] | None = None,
) -> list[PaperRecord]:
    """on_progress, if given, is called with short human-readable status
    strings as the search happens - lets a UI show what's taking a while
    instead of sitting on a blank screen during retries."""
    def notify(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,externalIds,url,isOpenAccess,openAccessPdf"
    }

    notify(f"Searching Semantic Scholar for \"{query}\"...")

    for attempt in range(max_retries):
        try:
            response = get_with_hard_timeout(url, params=params, timeout=20, hard_timeout=25)
        except requests.RequestException as exc:
            msg = f"Semantic Scholar network error, retrying... ({exc})"
            print(f"[semantic_scholar] {msg}")
            notify(msg)
            time.sleep(5 * (attempt + 1))
            continue

        if response.status_code == 429 or response.status_code >= 500:
            wait_time = 5 * (attempt + 1)
            msg = f"Semantic Scholar returned {response.status_code} - waiting {wait_time}s before retry"
            print(msg)
            notify(msg)
            time.sleep(wait_time)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            msg = f"Semantic Scholar request failed, skipping this term ({exc})"
            print(f"[semantic_scholar] {msg}")
            notify(msg)
            return []
        data = response.json()
        break
    else:
        msg = "Semantic Scholar still unavailable after retries - skipping this term"
        print(msg)
        notify(msg)
        return []

    papers = []
    for item in data.get("data", []):
        external_ids = item.get("externalIds") or {}
        paper_id = external_ids.get("DOI") or f"s2:{item['paperId']}"

        # item["url"] is the Semantic Scholar landing page, not a PDF - only
        # openAccessPdf.url actually points at a downloadable file.
        open_access_pdf = item.get("openAccessPdf") or {}
        pdf_url = open_access_pdf.get("url")

        paper = PaperRecord(
            id=paper_id,
            source="semantic_scholar",
            title=item.get("title", "Untitled"),
            authors=[a.get("name", "Unknown") for a in item.get("authors", [])],
            year=item.get("year"),
            url=pdf_url or item.get("url", ""),
            pdf_accessible=bool(pdf_url),
            subtopic_tags=[]
        )
        papers.append(paper)

    notify(f"Found {len(papers)} paper(s) on Semantic Scholar for \"{query}\"")
    return papers
