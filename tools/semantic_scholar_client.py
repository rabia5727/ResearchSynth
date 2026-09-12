"""Semantic Scholar search client — queries the Graph API and returns PaperRecords."""

import time
import requests

from state.schemas import PaperRecord


def search_semantic_scholar(query: str, max_results: int = 5, max_retries: int = 3) -> list[PaperRecord]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,externalIds,url,isOpenAccess,openAccessPdf"
    }

    for attempt in range(max_retries):
        response = requests.get(url, params=params)

        if response.status_code == 429:
            wait_time = 5 * (attempt + 1)
            print(f"Rate limited by Semantic Scholar. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue

        response.raise_for_status()
        data = response.json()
        break
    else:
        print("Semantic Scholar still rate-limited after retries. Skipping this query.")
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

    return papers