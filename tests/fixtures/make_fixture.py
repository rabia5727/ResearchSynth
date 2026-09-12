"""Generates a synthetic academic-paper-shaped PDF for testing Agent 2 (PDF
Reader) without needing a real downloaded paper. The content is fake but
shaped like a real paper - headers, a claim, a method, a result, a stated
limitation - and deliberately themed on the team's own demo question
(intermittent fasting + cognitive performance, see the plan section 9) so
running this script doubles as a rehearsal of the real thing.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def make_fixture_pdf(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=letter)

    story = [
        Paragraph("Fasting Windows and Working Memory: A Randomized Trial", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Abstract", styles["Heading1"]),
        Paragraph(
            "We investigate the effect of a 16:8 intermittent fasting schedule "
            "on working memory in healthy adults over a 6-week period.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Methodology", styles["Heading1"]),
        Paragraph(
            "42 participants were randomized to a 16:8 fasting schedule or an "
            "ad-libitum control diet for 6 weeks. Working memory was assessed "
            "weekly using an n-back task.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Results", styles["Heading1"]),
        Paragraph(
            "No significant difference in n-back accuracy was observed "
            "between groups at 6 weeks (p = 0.41). Fasting participants "
            "reported higher subjective alertness in the morning.",
            styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph("Limitations", styles["Heading1"]),
        Paragraph(
            "The 6-week duration may be too short to detect cognitive "
            "effects; prior work suggests changes emerge closer to 8-12 weeks.",
            styles["Normal"],
        ),
    ]
    doc.build(story)
    return path


if __name__ == "__main__":
    out = make_fixture_pdf(Path(__file__).parent / "sample_paper.pdf")
    print(f"Wrote fixture: {out}")
