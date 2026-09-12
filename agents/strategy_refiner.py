"""
Strategy Refiner Agent (Agent 5)

Responsibilities:
- Evaluate coverage (subtopics with >= 2 independent sources).
- Check unresolved tensions (resolved == False).
- Apply hard-stop conditions (Max cycles, diminishing returns).
- Call LLM to output a strictly typed RefinerDecision for Agent 1.
"""

from typing import Any, Dict, List, Literal
from pydantic import BaseModel

from tools.llm import generate_json

class RefinerDecision(BaseModel):
    decision: Literal["continue", "stop"]
    new_query_terms: List[str]
    papers_to_reexamine: List[str]
    rationale: str

def get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to safely get a value whether the state is a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def calculate_coverage(state: Any) -> float:
    """
    Calculate the current research coverage score (FR-10).
    A subtopic is covered if >= 2 independent papers are tagged with it.
    """
    subtopics = get_val(state, "subtopics", [])
    papers = get_val(state, "papers", [])

    if not subtopics:
        return 0.0

    covered_subtopics = 0
    for subtopic in subtopics:
        paper_ids = set()
        for p in papers:
            tags = get_val(p, "subtopic_tags", [])
            if subtopic in tags:
                paper_id = get_val(p, "id")
                if paper_id:
                    paper_ids.add(paper_id)

        if len(paper_ids) >= 2:
            covered_subtopics += 1

    return covered_subtopics / len(subtopics)

def check_max_cycles(state: Any) -> bool:
    cycle_n = get_val(state, "cycle_n", 0)
    max_cycles = get_val(state, "max_cycles", 2)
    return cycle_n >= max_cycles

def check_diminishing_returns(state: Any) -> bool:
    strategy_log = get_val(state, "strategy_log", [])
    if not strategy_log:
        return False

    latest_entry = strategy_log[-1]
    papers_yielded = get_val(latest_entry, "papers_yielded", 0)
    new_unique = get_val(latest_entry, "new_unique_findings", 0)

    if papers_yielded > 0 and (new_unique / papers_yielded) < 0.10:
        return True
    return False

def get_unresolved_tensions(state: Any) -> list:
    """
    Filters for tensions where resolved is False, per Agent 3's logic.
    """
    tensions = get_val(state, "tensions", [])
    return [t for t in tensions if not get_val(t, "resolved", False)]

def strategy_refiner_node(state: Any) -> Dict[str, Any]:
    """
    Main LangGraph node function for the Strategy Refiner.
    """
    coverage_score = calculate_coverage(state)
    unresolved_tensions = get_unresolved_tensions(state)
    
    stop_reason = None
    
    # 1. Check Hard Stops
    if check_max_cycles(state):
        stop_reason = "Maximum cycle limit reached."
    elif check_diminishing_returns(state):
        stop_reason = "Diminishing returns: new-unique-findings ratio < 10%."

    # 2. Handle Stop Condition
    if stop_reason:
        decision_dict = {
            "decision": "stop",
            "new_query_terms": [],
            "papers_to_reexamine": [], 
            "rationale": stop_reason
        }
        return {
            "coverage_score": coverage_score,
            "stop_reason": stop_reason,
            "refiner_decision": decision_dict
        }

    # 3. Handle Continue Condition (LLM Call)
    subtopics = get_val(state, "subtopics", [])
    strategy_log = get_val(state, "strategy_log", [])
    
    prompt = f"""
    You are the Strategy Refiner for ResearchSynth.
    Analyze the current state to refine the search strategy for the next cycle.
    
    Current Coverage Score: {coverage_score}
    Subtopics: {subtopics}
    Unresolved Tensions: {unresolved_tensions}
    Search History: {strategy_log}
    
    Output a JSON matching the RefinerDecision schema. Provide targeted new_query_terms 
    to address coverage gaps and unresolved tensions.
    """
    
    # generate_json returns a validated RefinerDecision instance, not a dict
    # (see tools/llm.py) - pass the model class itself, not its json schema.
    response = generate_json(prompt, RefinerDecision)

    return {
        "coverage_score": coverage_score,
        "refiner_decision": {
            "decision": response.decision,
            "new_query_terms": response.new_query_terms,
            "papers_to_reexamine": response.papers_to_reexamine,
            "rationale": response.rationale
        }
    }