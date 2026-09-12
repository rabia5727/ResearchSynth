import unittest
from unittest.mock import patch

from agents.strategy_refiner import (
    calculate_coverage,
    check_max_cycles,
    check_diminishing_returns,
    get_unresolved_tensions,
    strategy_refiner_node,
    RefinerDecision,
)

class TestStrategyRefiner(unittest.TestCase):

    def test_calculate_coverage(self):
        state = {
            "subtopics": ["Subtopic A", "Subtopic B"],
            "papers": [
                {"id": "p1", "subtopic_tags": ["Subtopic A"]},
                {"id": "p2", "subtopic_tags": ["Subtopic A"]},
                {"id": "p3", "subtopic_tags": ["Subtopic B"]}
            ]
        }
        coverage = calculate_coverage(state)
        self.assertEqual(coverage, 0.5)

    def test_check_max_cycles(self):
        self.assertFalse(check_max_cycles({"cycle_n": 1, "max_cycles": 2}))
        self.assertTrue(check_max_cycles({"cycle_n": 2, "max_cycles": 2}))

    def test_check_diminishing_returns(self):
        state_stop = {
            "strategy_log": [{"papers_yielded": 10, "new_unique_findings": 0}]
        }
        self.assertTrue(check_diminishing_returns(state_stop))

        state_continue = {
            "strategy_log": [{"papers_yielded": 10, "new_unique_findings": 2}]
        }
        self.assertFalse(check_diminishing_returns(state_continue))

    def test_get_unresolved_tensions(self):
        state = {
            "tensions": [
                {"id": 1, "resolved": True},
                {"id": 2, "resolved": False},
                {"id": 3}  # Defaults to False
            ]
        }
        unresolved = get_unresolved_tensions(state)
        self.assertEqual(len(unresolved), 2)
        ids = [t.get("id") for t in unresolved]
        self.assertIn(2, ids)
        self.assertIn(3, ids)

    def test_strategy_refiner_node_hard_stop(self):
        state = {
            "cycle_n": 3,
            "max_cycles": 3,
            "subtopics": [],
            "papers": []
        }
        result = strategy_refiner_node(state)
        
        self.assertEqual(result["refiner_decision"]["decision"], "stop")
        self.assertEqual(result["stop_reason"], "Maximum cycle limit reached.")
        self.assertEqual(result["refiner_decision"]["new_query_terms"], [])

    @patch("agents.strategy_refiner.generate_json")
    def test_strategy_refiner_node_continue(self, mock_generate_json):
        mock_generate_json.return_value = RefinerDecision(
            decision="continue",
            new_query_terms=["intermittent fasting", "longevity"],
            papers_to_reexamine=["p4", "p5"],
            rationale="Testing LLM routing.",
        )
        
        state = {
            "cycle_n": 1,
            "max_cycles": 3,
            "subtopics": ["fasting"],
            "papers": [],
            "tensions": [],
            "strategy_log": [{"papers_yielded": 10, "new_unique_findings": 5}]
        }
        
        result = strategy_refiner_node(state)
        
        self.assertEqual(result["refiner_decision"]["decision"], "continue")
        self.assertEqual(result["refiner_decision"]["new_query_terms"], ["intermittent fasting", "longevity"])
        self.assertEqual(result["refiner_decision"]["papers_to_reexamine"], ["p4", "p5"])
        mock_generate_json.assert_called_once()

if __name__ == '__main__':
    unittest.main()