"""arXiv search client — queries the arXiv Atom API and returns PaperRecords."""

import urllib.parse
import feedparser
import requests

from state.schemas import PaperRecord


def search_arxiv(query: str, max_results: int = 5) -> list[PaperRecord]:
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    url = base_url + "?" + urllib.parse.urlencode(params)

    response = requests.get(url)
    response.raise_for_status()
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

    return papers