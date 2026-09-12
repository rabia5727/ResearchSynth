"""Stand-alone smoke test for Agent 2 (PDF Reader).

Proves the download/parse/extract wiring works end to end using a synthetic
fixture paper - no real download, no real API key needed if LLM_PROVIDER=mock
(the default). Run it from the repo root:

    python scripts/demo_pdf_reader.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.pdf_reader import extract_findings  # noqa: E402
from state.schemas import PaperRecord  # noqa: E402
from tests.fixtures.make_fixture import make_fixture_pdf  # noqa: E402


def main() -> None:
    fixture_path = make_fixture_pdf(REPO_ROOT / "tests" / "fixtures" / "sample_paper.pdf")

    paper = PaperRecord(
        id="fixture:001",
        source="arxiv",
        title="Fasting Windows and Working Memory: A Randomized Trial",
        authors=["A. Researcher"],
        year=2024,
        url="local://fixture",
        pdf_accessible=True,
    )

    findings = extract_findings(paper, str(fixture_path))

    print(f"\nExtracted {len(findings)} finding(s) from {paper.title!r}:\n")
    for f in findings:
        print(f"- [{f.section_source}] {f.claim}")
        print(f"    method={f.method!r} dataset={f.dataset!r} metric={f.metric!r} value={f.value!r}")
        print(f"    limitations={f.limitations!r}\n")


if __name__ == "__main__":
    main()
