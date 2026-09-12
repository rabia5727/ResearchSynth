"""Stand-alone smoke test for the full adaptive loop (graph.py).

Contradiction Detector needs a real GEMINI_API_KEY + PINECONE_API_KEY to run
for real (it's the one agent that doesn't support LLM_PROVIDER=mock - see
the plan). This script patches it out with a no-op so the rest of the loop
- Searcher, PDF Reader, Synthesis Writer, Strategy Refiner, and the
cycle-to-cycle wiring itself - can be smoke-tested with LLM_PROVIDER=mock
and zero external accounts.

Once real keys are available for every agent, run `python graph.py`
directly instead - that exercises the real Contradiction Detector too.

    python scripts/demo_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    with patch("graph.detect_contradictions", return_value={"tensions": []}):
        import graph

        result = graph.run(
            "What is the effect of intermittent fasting on cognitive performance?"
        )

    print(f"Stopped after {result.cycle_n} cycle(s): {result.stop_reason or 'refiner decided to stop'}")
    print(f"Coverage score: {result.coverage_score:.2f}")
    print(f"Papers: {len(result.papers)} | Findings: {len(result.findings)} | Tensions: {len(result.tensions)}")
    print(f"\nStrategy log ({len(result.strategy_log)} entries):")
    for entry in result.strategy_log:
        print(
            f"  cycle {entry.cycle_n}: terms={entry.query_terms} "
            f"papers={entry.papers_yielded} new_findings={entry.new_unique_findings} "
            f"-> {entry.refiner_decision} ({entry.rationale[:80] if entry.rationale else ''})"
        )

    print("\n--- Final report ---\n")
    print(result.latest_report or "(no report generated)")


if __name__ == "__main__":
    main()
