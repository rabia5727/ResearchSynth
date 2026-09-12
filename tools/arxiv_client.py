"""arXiv search client — queries the arXiv Atom API and returns PaperRecords."""

import time
import urllib.parse
from typing import Callable

import feedparser
import requests

from state.schemas import PaperRecord


def search_arxiv(
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

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    url = base_url + "?" + urllib.parse.urlencode(params)

    notify(f"Searching arXiv for \"{query}\"...")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=20)
        except requests.RequestException as exc:
            msg = f"arXiv network error, retrying... ({exc})"
            print(f"[arxiv] {msg}")
            notify(msg)
            time.sleep(5 * (attempt + 1))
            continue

        if response.status_code == 429 or response.status_code >= 500:
            wait_time = 5 * (attempt + 1)
            msg = f"arXiv returned {response.status_code} - waiting {wait_time}s before retry"
            print(msg)
            notify(msg)
            time.sleep(wait_time)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            msg = f"arXiv request failed, skipping this term ({exc})"
            print(f"[arxiv] {msg}")
            notify(msg)
            return []
        break
    else:
        msg = "arXiv still unavailable after retries - skipping this term"
        print(msg)
        notify(msg)
        return []

    feed = feedparser.parse(response.content)

    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        year = None
        if hasattr(entry, "published"):
            year = int(entry.published[:4])

        paper = PaperRecord(
            id=f"arxiv:{arxiv_id}",
            source="arxiv",
            title=entry.title.replace("\n", " ").strip(),
            authors=[author.name for author in entry.authors],
            year=year,
            # entry.link is the abstract page (/abs/...) - arXiv's PDF is
            # always at this convention, not something feedparser exposes
            # directly as a clean field.
            url=f"https://arxiv.org/pdf/{arxiv_id}",
            pdf_accessible=True,
            subtopic_tags=[]
        )
        papers.append(paper)

    notify(f"Found {len(papers)} paper(s) on arXiv for \"{query}\"")
    return papers
