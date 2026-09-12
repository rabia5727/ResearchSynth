"""LangGraph wiring for the ResearchSynth adaptive research loop.

Searcher -> PDF Reader -> Contradiction Detector -> Synthesis Writer ->
Strategy Refiner -> (loop back to Searcher with a refined strategy, or stop).

Each agent module (agents/*.py) is a standalone, independently-testable
function - none of them know about each other or about LangGraph. This file
is the one place that owns the bookkeeping the plan assigns to "whoever
wires the graph": merging new papers/findings into CycleState, deduping
repeat tensions across cycles, and building the StrategyMemoryEntry the
Strategy Refiner reads to decide whether coverage is good enough yet.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.contradiction_detector import detect_contradictions
from agents.pdf_reader import extract_findings
from agents.searcher import decompose_query, searcher_node
from agents.strategy_refiner import strategy_refiner_node
from agents.synthesis_writer import SynthesisWriter
from state.schemas import CycleState, PaperRecord, StrategyMemoryEntry

_synthesis_writer = SynthesisWriter()
_SOURCES = ["arxiv", "semantic_scholar"]  # pubmed is a post-hackathon stretch


class GraphState(TypedDict):
    """The graph's own wrapper around CycleState.

    CycleState (state/schemas.py) is the frozen inter-agent contract and
    stays exactly as every agent expects it. Anything that's purely
    orchestration bookkeeping - not something an individual agent reads or
    writes - lives here instead, so the shared schema doesn't get polluted
    with LangGraph-specific plumbing.
    """

    cycle: CycleState
    pending_refiner_decision: dict | None  # None until cycle 1's Refiner runs
    current_query_terms: list[str]  # terms Searcher just used this pass


def decompose_step(state: GraphState) -> dict:
    """Runs exactly once, before the loop - see plan FR-02. Populates
    CycleState.subtopics so Searcher has something to search for on cycle 1.
    """
    cycle = state["cycle"]
    subtopics = decompose_query(cycle.query)
    return {"cycle": cycle.model_copy(update={"subtopics": subtopics})}


def searcher_step(state: GraphState) -> dict:
    """Node 1 - search with the current strategy (initial subtopics on
    cycle 1, the Refiner's new_query_terms on every cycle after)."""
    cycle = state["cycle"]
    refiner_decision = state["pending_refiner_decision"]

    result = searcher_node(cycle, refiner_decision=refiner_decision)
    new_papers: list[PaperRecord] = result.get("new_papers", [])

    query_terms = (
        refiner_decision["new_query_terms"] if refiner_decision else cycle.subtopics
    )

    updated_cycle = cycle.model_copy(
        update={
            "papers": cycle.papers + new_papers,
            "cycle_n": cycle.cycle_n + 1,
        }
    )
    return {"cycle": updated_cycle, "current_query_terms": query_terms}


def pdf_reader_step(state: GraphState) -> dict:
    """Node 2 - extract structured findings for every paper that doesn't
    have any yet, plus re-run extraction (bypassing the cache) for any
    paper the Refiner flagged for re-examination.

    Also builds this cycle's StrategyMemoryEntry - papers_yielded and
    new_unique_findings are known here; refiner_decision/rationale are
    filled in as a placeholder and overwritten by refiner_step once the
    Refiner actually decides (see strategy_refiner_node's own
    check_diminishing_returns, which reads exactly these two fields off
    the last log entry).
    """
    cycle = state["cycle"]
    refiner_decision = state["pending_refiner_decision"]
    reexamine_ids = set(refiner_decision["papers_to_reexamine"]) if refiner_decision else set()

    findings = list(cycle.findings)
    already_covered = {f.paper_id for f in findings}
    papers_by_id = {p.id: p for p in cycle.papers}

    new_paper_count = 0
    new_finding_count = 0

    # New papers this cycle - never processed before
    for paper in cycle.papers:
        if paper.id in already_covered and paper.id not in reexamine_ids:
            continue
        if paper.id in reexamine_ids:
            continue  # handled separately below, with force_refresh + focus_note
        new_paper_count += 1
        extracted = extract_findings(paper)
        findings.extend(extracted)
        new_finding_count += len(extracted)

    # Re-examined papers - replace their old findings with a fresh pass
    for paper_id in reexamine_ids:
        paper = papers_by_id.get(paper_id)
        if paper is None:
            continue
        findings = [f for f in findings if f.paper_id != paper_id]
        extracted = extract_findings(
            paper, force_refresh=True, focus_note=refiner_decision.get("rationale")
        )
        findings.extend(extracted)
        new_finding_count += len(extracted)

    log_entry = StrategyMemoryEntry(
        cycle_n=cycle.cycle_n,
        query_terms=state["current_query_terms"],
        sources_queried=_SOURCES,
        papers_yielded=new_paper_count,
        new_unique_findings=new_finding_count,
        refiner_decision="continue",  # placeholder - refiner_step overwrites this
        rationale="",
    )

    updated_cycle = cycle.model_copy(
        update={"findings": findings, "strategy_log": cycle.strategy_log + [log_entry]}
    )
    return {"cycle": updated_cycle}


def contradiction_step(state: GraphState) -> dict:
    """Node 3 - detect_contradictions re-scans all findings each cycle, so
    dedupe against tensions we already recorded in an earlier cycle before
    appending (same pair, same relation, doesn't need to be added twice)."""
    cycle = state["cycle"]
    result = detect_contradictions(cycle)
    candidate_tensions = result.get("tensions", [])

    seen_pairs = {
        tuple(sorted((t.finding_a_id, t.finding_b_id))) for t in cycle.tensions
    }
    genuinely_new = []
    for tension in candidate_tensions:
        pair_key = tuple(sorted((tension.finding_a_id, tension.finding_b_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        genuinely_new.append(tension)

    updated_cycle = cycle.model_copy(
        update={"tensions": cycle.tensions + genuinely_new}
    )
    return {"cycle": updated_cycle}


def synthesis_step(state: GraphState) -> dict:
    """Node 4 - regenerate the report every cycle; the final cycle's
    version is the deliverable. A grounding failure shouldn't crash the
    whole loop, so on ValueError we log it and keep the previous report."""
    cycle = state["cycle"]
    try:
        report = _synthesis_writer.synthesize(cycle)
    except ValueError as exc:
        print(f"[graph] synthesis failed this cycle, keeping previous report: {exc}")
        return {}

    return {"cycle": cycle.model_copy(update={"latest_report": report})}


def refiner_step(state: GraphState) -> dict:
    """Node 5 - the core differentiator. Evaluates coverage, decides
    continue/stop, and (on continue) proposes the next cycle's strategy."""
    cycle = state["cycle"]
    result = strategy_refiner_node(cycle)
    decision = result["refiner_decision"]

    log = list(cycle.strategy_log)
    if log:
        log[-1] = log[-1].model_copy(
            update={"refiner_decision": decision["decision"], "rationale": decision["rationale"]}
        )

    updated_cycle = cycle.model_copy(
        update={
            "coverage_score": result["coverage_score"],
            "stop_reason": result.get("stop_reason"),
            "strategy_log": log,
        }
    )
    return {"cycle": updated_cycle, "pending_refiner_decision": decision}


def _should_continue(state: GraphState) -> str:
    decision = state["pending_refiner_decision"]
    return "searcher" if decision and decision["decision"] == "continue" else END


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("decompose", decompose_step)
    graph.add_node("searcher", searcher_step)
    graph.add_node("pdf_reader", pdf_reader_step)
    graph.add_node("contradiction_detector", contradiction_step)
    graph.add_node("synthesis_writer", synthesis_step)
    graph.add_node("strategy_refiner", refiner_step)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "searcher")
    graph.add_edge("searcher", "pdf_reader")
    graph.add_edge("pdf_reader", "contradiction_detector")
    graph.add_edge("contradiction_detector", "synthesis_writer")
    graph.add_edge("synthesis_writer", "strategy_refiner")
    graph.add_conditional_edges("strategy_refiner", _should_continue, {"searcher": "searcher", END: END})

    return graph.compile()


def run(query: str, max_cycles: int = 2) -> CycleState:
    """Convenience entry point: run the full adaptive loop for one query
    and return the final CycleState (report, tensions, strategy_log, etc.)."""
    app = build_graph()
    initial_state: GraphState = {
        "cycle": CycleState(query=query, max_cycles=max_cycles),
        "pending_refiner_decision": None,
        "current_query_terms": [],
    }
    final_state = app.invoke(initial_state)
    return final_state["cycle"]


if __name__ == "__main__":
    result = run("What is the effect of intermittent fasting on cognitive performance?")
    print(f"\nStopped after {result.cycle_n} cycle(s): {result.stop_reason or 'refiner decided to stop'}")
    print(f"Coverage score: {result.coverage_score:.2f}")
    print(f"Papers: {len(result.papers)} | Findings: {len(result.findings)} | Tensions: {len(result.tensions)}")
    print("\n--- Final report ---\n")
    print(result.latest_report or "(no report generated)")
