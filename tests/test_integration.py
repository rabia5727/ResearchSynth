import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents')))

# Attempt to load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

from state.schemas import CycleState, ExtractedFinding, PaperRecord
from agents.contradiction_detector import detect_contradictions


class TestContradictionDetectorIntegration(unittest.TestCase):
    def setUp(self):
        # We need real API keys to run this integration test
        if not os.environ.get("GEMINI_API_KEY") or not os.environ.get("PINECONE_API_KEY"):
            self.skipTest("Real API keys (GEMINI_API_KEY, PINECONE_API_KEY) are required for this integration test.")

    def test_real_contradictions(self):
        # Create mock papers with the same subtopic
        papers = [
            PaperRecord(
                id="arxiv:1001",
                source="arxiv",
                title="Positive effects of fasting",
                url="http://example.com/1",
                subtopic_tags=["fasting_cognitive"]
            ),
            PaperRecord(
                id="arxiv:1002",
                source="arxiv",
                title="Null effects of fasting",
                url="http://example.com/2",
                subtopic_tags=["fasting_cognitive"]
            )
        ]

        # Create explicit findings that contradict to test the LLM and vector search
        findings = [
            ExtractedFinding(
                id="finding_1",
                paper_id="arxiv:1001",
                claim="Intermittent fasting significantly improves cognitive function and memory in adult mice over an 8-week period.",
                method="8-week controlled diet in adult mice",
                metric="Memory test scores",
                section_source="results"
            ),
            ExtractedFinding(
                id="finding_2",
                paper_id="arxiv:1002",
                claim="Intermittent fasting produces no significant changes in cognitive function or memory in adult mice over an 8-week period.",
                method="8-week controlled diet in adult mice",
                metric="Memory test scores",
                section_source="results"
            )
        ]

        # Construct the state (cycle_n=1 triggers the clear_namespace logic)
        state = CycleState(
            query="Intermittent fasting cognitive effects",
            cycle_n=1,
            papers=papers,
            findings=findings
        )

        # Run the actual detector
        result = detect_contradictions(state)

        # Assertions
        self.assertIn("tensions", result, "The result must contain a 'tensions' key.")
        tensions = result["tensions"]
        
        # We expect at least one tension between the strictly contradicting claims
        self.assertGreaterEqual(len(tensions), 1, "At least one contradiction should be found.")

        # Let's verify the contents of the detected tension
        tension = tensions[0]
        # Depending on order, it could be (finding_1, finding_2) or (finding_2, finding_1)
        valid_pairs = [("finding_1", "finding_2"), ("finding_2", "finding_1")]
        self.assertIn((tension.finding_a_id, tension.finding_b_id), valid_pairs)
        self.assertEqual(tension.relation, "contradicts", "The LLM should classify direct opposites as 'contradicts'.")
        self.assertFalse(tension.resolved, "Tension should default to unresolved.")

        print("\n--- Detected Tension ---")
        print(f"Finding A: {tension.finding_a_id}")
        print(f"Finding B: {tension.finding_b_id}")
        print(f"Relation: {tension.relation}")
        print(f"Score: {tension.score}")
        print(f"Explanation: {tension.explanation}")


if __name__ == '__main__':
    unittest.main()
