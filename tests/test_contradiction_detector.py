import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents')))

os.environ["PINECONE_API_KEY"] = "dummy_key"
os.environ["GEMINI_API_KEY"] = "dummy_key"

import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock Pinecone to prevent network calls during import
sys.modules['pinecone'] = MagicMock()
sys.modules['pinecone'].Pinecone = MagicMock()

from state.schemas import CycleState, ExtractedFinding, PaperRecord, ContradictionPair
from agents.contradiction_detector import detect_contradictions

class TestContradictionDetector(unittest.TestCase):
    @patch('agents.contradiction_detector.vector_store')
    @patch('agents.contradiction_detector.genai.Client')
    def test_detect_contradictions(self, MockClient, mock_vector_store):
        # Setup mock client
        mock_client_instance = MagicMock()
        MockClient.return_value = mock_client_instance
        
        # Setup mock embeddings
        mock_embed_response = MagicMock()
        mock_embed_response.embeddings = [
            MagicMock(values=[0.1] * 3072),
            MagicMock(values=[0.2] * 3072)
        ]
        mock_client_instance.models.embed_content.return_value = mock_embed_response
        
        # Setup mock generated content for tension classification
        mock_generate_response = MagicMock()
        mock_generate_response.parsed = ContradictionPair(
            finding_a_id="f1",
            finding_b_id="f2",
            relation="contradicts",
            score=0.9,
            explanation="Different methodologies resulted in opposing outcomes."
        )
        mock_client_instance.models.generate_content.return_value = mock_generate_response

        # Setup mock vector store search results
        # We need it to return a match above the threshold 0.75
        def mock_search(query_vector, subtopic, top_k):
            # For finding 1, return finding 2 as a match
            if query_vector == [0.1] * 3072:
                return {
                    "matches": [
                        {"id": "f2", "score": 0.85, "metadata": {"claim": "Claim 2", "method": "Method 2", "metric": "Metric 2"}}
                    ]
                }
            return {"matches": []}
            
        mock_vector_store.search_within_subtopic.side_effect = mock_search

        # Construct input state
        state = CycleState(
            query="test query",
            papers=[
                PaperRecord(id="p1", source="arxiv", title="Paper 1", url="url1", subtopic_tags=["tech"]),
                PaperRecord(id="p2", source="arxiv", title="Paper 2", url="url2", subtopic_tags=["tech"])
            ],
            findings=[
                ExtractedFinding(id="f1", paper_id="p1", claim="Claim 1", method="Method 1", metric="Metric 1", section_source="results"),
                ExtractedFinding(id="f2", paper_id="p2", claim="Claim 2", method="Method 2", metric="Metric 2", section_source="results")
            ]
        )

        result = detect_contradictions(state)
        
        self.assertIn("tensions", result)
        tensions = result["tensions"]
        self.assertEqual(len(tensions), 1)
        self.assertEqual(tensions[0].finding_a_id, "f1")
        self.assertEqual(tensions[0].finding_b_id, "f2")
        self.assertEqual(tensions[0].relation, "contradicts")
        self.assertEqual(tensions[0].resolved, False)

if __name__ == '__main__':
    unittest.main()
