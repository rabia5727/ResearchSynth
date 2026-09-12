"""ResearchSynth - Streamlit demo UI.

Runs the full adaptive loop (graph.py) end to end, in-process, and shows
each cycle live: what changed in the search strategy, what was found, and
why the Strategy Refiner decided to continue or stop. This is the
"cycle 1 vs. cycle 2" demo surface from the implementation plan, section 9.

No backend, no WebSocket - graph.app.stream() is consumed directly in this
script (see the plan's scope cuts, section 0).
"""
from __future__ import annotations

import html
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from agents.contradiction_detector import detect_contradictions as _real_detect_contradictions  # noqa: E402
from graph import GraphState, build_graph  # noqa: E402
from state.schemas import CycleState  # noqa: E402

DEMO_QUERY = "What is the effect of intermittent fasting on cognitive performance?"

_RELATION = {
    "contradicts": ("Contradicts", "contradicts"),
    "partial_consensus": ("Partial consensus", "consensus"),
    "methodological_divergence": ("Methodological divergence", "divergence"),
}


# ---------------------------------------------------------------------------
# Styling - IBM Plex everywhere, teal/paper palette, no emoji
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #F5F6F3; --surface: #FFFFFF; --surface-2: #EDF0EA;
            --ink: #1C2321; --ink-muted: #5B6560; --border: #DCE2DD;
            --accent: #2F6F62; --accent-ink: #1F4E44; --accent-soft: #E4EEEA;
            --warn: #B5533C; --warn-soft: #F5E6E0;
        }
        html, body, [class*="css"], .stMarkdown, .stTextInput input { font-family: 'IBM Plex Sans', sans-serif !important; }
        h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif !important; color: var(--ink) !important; letter-spacing: -0.01em; }
        .rs-eyebrow {
            display:inline-flex; align-items:center; gap:8px;
            font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:.06em; text-transform:uppercase;
            color: var(--accent-ink); background: var(--accent-soft); border:1px solid var(--border);
            padding:4px 10px; border-radius:100px; margin-bottom:6px;
        }
        .rs-subtitle { color: var(--ink-muted); font-size: 15.5px; max-width: 68ch; margin-bottom: 4px; }
        .stButton>button {
            font-family:'IBM Plex Sans', sans-serif; font-weight:600; border-radius:8px;
            border:1px solid var(--accent); background: var(--accent); color:#fff;
        }
        .stButton>button:hover { background: var(--accent-ink); border-color: var(--accent-ink); color:#fff; }
        [data-testid="stMetricValue"] { font-family:'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; color: var(--ink); }
        [data-testid="stMetricLabel"] { font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:.05em; text-transform:uppercase; color: var(--ink-muted) !important; }
        .rs-pill { display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:11.5px; padding:2px 9px; border-radius:100px; white-space:nowrap; }
        .rs-pill-contradicts { background: var(--warn-soft); color: var(--warn); }
        .rs-pill-consensus { background: var(--accent-soft); color: var(--accent-ink); }
        .rs-pill-divergence { background: var(--surface-2); color: var(--ink-muted); border:1px solid var(--border); }
        .rs-pill-continue { background: var(--accent-soft); color: var(--accent-ink); }
        .rs-pill-stop { background: var(--surface-2); color: var(--ink-muted); border:1px solid var(--border); }
        .rs-table-wrap { overflow-x: auto; border:1px solid var(--border); border-radius:10px; }
        .rs-table { width:100%; border-collapse: collapse; font-size: 13.6px; background: var(--surface); }
        .rs-table th { text-align:left; font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:.05em; text-transform:uppercase; color: var(--ink-muted); border-bottom:1px solid var(--border); padding:9px 12px; background: var(--surface-2); }
        .rs-table td { padding:10px 12px; border-bottom:1px solid var(--border); vertical-align:top; color: var(--ink); }
        .rs-table tr:last-child td { border-bottom:none; }
        .rs-caption { color: var(--ink-muted); font-size: 13.3px; margin-top: -6px; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _pill(text: str, kind: str) -> str:
    return f'<span class="rs-pill rs-pill-{kind}">{html.escape(text)}</span>'


def _relation_pill(relation: str) -> str:
    label, kind = _RELATION.get(relation, (relation, "divergence"))
    return _pill(label, kind)


def _decision_pill(decision: str) -> str:
    return _pill(decision.capitalize(), "continue" if decision == "continue" else "stop")


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------

def _render_header() -> None:
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    st.markdown('<span class="rs-eyebrow">Adaptive literature reviewer</span>', unsafe_allow_html=True)
    st.title("ResearchSynth")
    st.markdown(
        '<p class="rs-subtitle">Searches academic papers, extracts structured findings, finds '
        'where they agree or disagree, and refines its own search strategy when coverage is '
        'thin - instead of stopping after one pass.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(_pill(f"LLM_PROVIDER: {provider}", "continue" if provider != "mock" else "stop"), unsafe_allow_html=True)
    st.write("")


def _render_form() -> tuple[str, int, bool]:
    with st.form("run_form"):
        query = st.text_input("Research question", value=DEMO_QUERY)
        max_cycles = st.slider("Maximum cycles", min_value=1, max_value=3, value=2)
        submitted = st.form_submit_button("Run research", type="primary")
    return query, max_cycles, submitted


@st.cache_resource
def _get_app():
    return build_graph()


def _run_and_render(query: str, max_cycles: int) -> None:
    have_pinecone = bool(os.environ.get("PINECONE_API_KEY"))
    have_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    if have_pinecone and have_gemini:
        ctx = nullcontext()
    else:
        st.info(
            "Contradiction Detector needs a real GEMINI_API_KEY and PINECONE_API_KEY to run - "
            "it's the one agent without a mock path (see the implementation plan). "
            "Skipping tension detection for this run."
        )
        ctx = patch("graph.detect_contradictions", return_value={"tensions": []})

    app = _get_app()
    initial_state: GraphState = {
        "cycle": CycleState(query=query, max_cycles=max_cycles),
        "pending_refiner_decision": None,
        "current_query_terms": [],
    }

    current: dict = dict(initial_state)
    cycle_status = None

    st.divider()
    st.subheader("Run")

    try:
        with ctx:
            for chunk in app.stream(initial_state, stream_mode="updates"):
                (node_name, update), = chunk.items()
                if update:  # LangGraph reports a node's {} return as None here
                    current.update(update)
                cycle: CycleState = current["cycle"]

                if node_name == "decompose":
                    st.markdown(f"**Decomposed into subtopics:** {', '.join(cycle.subtopics)}")

                elif node_name == "searcher":
                    cycle_status = st.status(f"Cycle {cycle.cycle_n}", expanded=True)
                    terms = ", ".join(current["current_query_terms"])
                    cycle_status.write(f"Searching with: {terms}")
                    cycle_status.write(f"Papers so far: {len(cycle.papers)}")

                elif node_name == "pdf_reader":
                    cycle_status.write(f"Findings extracted - {len(cycle.findings)} total so far")

                elif node_name == "contradiction_detector":
                    cycle_status.write(f"Tensions detected - {len(cycle.tensions)} total so far")

                elif node_name == "synthesis_writer":
                    msg = "Synthesis written" if cycle.latest_report else "Synthesis skipped (insufficient grounded evidence this cycle)"
                    cycle_status.write(msg)

                elif node_name == "strategy_refiner":
                    decision = current["pending_refiner_decision"]
                    cycle_status.write(f"Coverage: {cycle.coverage_score:.0%}")
                    cycle_status.write(f"Refiner decision: {decision['decision']} - {decision['rationale']}")
                    cycle_status.update(label=f"Cycle {cycle.cycle_n} - {decision['decision']}", state="complete")
    except Exception as exc:  # surface it in the UI rather than a bare traceback
        st.error(f"The run stopped early: {exc}")

    _render_results(current["cycle"])


def _render_results(cycle: CycleState) -> None:
    st.divider()
    st.subheader("Results")

    cols = st.columns(4)
    cols[0].metric("Papers", len(cycle.papers))
    cols[1].metric("Findings", len(cycle.findings))
    cols[2].metric("Tensions", len(cycle.tensions))
    cols[3].metric("Coverage", f"{cycle.coverage_score:.0%}")

    st.markdown("#### Strategy log")
    st.markdown(
        '<p class="rs-caption">The adaptive part - watch the search terms change between cycles.</p>',
        unsafe_allow_html=True,
    )
    if cycle.strategy_log:
        rows = "".join(
            f"<tr><td>{e.cycle_n}</td><td>{html.escape(', '.join(e.query_terms))}</td>"
            f"<td>{e.papers_yielded}</td><td>{e.new_unique_findings}</td>"
            f"<td>{_decision_pill(e.refiner_decision)}</td>"
            f"<td>{html.escape(e.rationale or '')}</td></tr>"
            for e in cycle.strategy_log
        )
        st.markdown(
            '<div class="rs-table-wrap"><table class="rs-table"><thead><tr>'
            "<th>Cycle</th><th>Query terms</th><th>Papers</th><th>New findings</th>"
            f"<th>Decision</th><th>Rationale</th></tr></thead><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No cycles recorded yet.")

    if cycle.tensions:
        st.markdown("#### Tension map")
        rows = "".join(
            f"<tr><td>{html.escape(t.finding_a_id)}</td><td>{html.escape(t.finding_b_id)}</td>"
            f"<td>{_relation_pill(t.relation)}</td><td>{t.score:.2f}</td>"
            f"<td>{html.escape(t.explanation)}</td></tr>"
            for t in cycle.tensions
        )
        st.markdown(
            '<div class="rs-table-wrap"><table class="rs-table"><thead><tr>'
            "<th>Finding A</th><th>Finding B</th><th>Relation</th><th>Score</th>"
            f"<th>Explanation</th></tr></thead><tbody>{rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Final report")
    if cycle.latest_report:
        with st.container(border=True):
            st.markdown(cycle.latest_report)
    else:
        st.info("No report was generated - insufficient grounded evidence to synthesize.")


def main() -> None:
    st.set_page_config(page_title="ResearchSynth", layout="wide")
    _inject_css()
    _render_header()
    query, max_cycles, submitted = _render_form()

    if submitted:
        if not query.strip():
            st.warning("Enter a research question first.")
        else:
            _run_and_render(query.strip(), max_cycles)


if __name__ == "__main__":
    main()
