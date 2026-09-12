import pytest
from unittest.mock import patch
from state.schemas import CycleState, ExtractedFinding, PaperRecord, ContradictionPair
from agents.synthesis_writer import SynthesisWriter, SynthesisResponse

@pytest.fixture
def base_state():
    return CycleState(
        query="What is the effect of intermittent fasting on cognitive performance?",
        subtopics=["Attention", "Memory"],
        cycle_n=1,
        papers=[
            PaperRecord(
                id="P001", source="arxiv", title="Fasting and Attention", authors=["Alice"],
                year=2023, url="url", pdf_accessible=True, subtopic_tags=["Attention"]
            ),
            PaperRecord(
                id="P002", source="pubmed", title="Memory in Fasting", authors=["Bob"],
                year=2024, url="url2", pdf_accessible=True, subtopic_tags=["Memory"]
            ),
            PaperRecord(
                id="P003", source="pubmed", title="Other Fasting Details", authors=["Charlie"],
                year=2025, url="url3", pdf_accessible=True, subtopic_tags=[]
            ),
            PaperRecord(
                id="P004", source="pubmed", title="Attention Fasting", authors=["Dave"],
                year=2025, url="url4", pdf_accessible=True, subtopic_tags=["Attention"]
            )
        ],
        findings=[
            ExtractedFinding(
                id="F001", paper_id="P001", claim="Improved attention.", method="RCT",
                dataset="D1", metric="Score", value="High", limitations="None", section_source="Results"
            ),
            ExtractedFinding(
                id="F002", paper_id="P002", claim="Decreased memory.", method="Obs",
                dataset=None, metric=None, value=None, limitations="Small N", section_source="Conclusion"
            ),
            ExtractedFinding(
                id="F003", paper_id="P003", claim="Some unassigned finding.", method="Survey",
                dataset=None, metric=None, value=None, limitations="Self-reported", section_source="Discussion"
            ),
            ExtractedFinding(
                id="F004", paper_id="P004", claim="Attention unchanged.", method="RCT",
                dataset=None, metric=None, value=None, limitations="None", section_source="Results"
            )
        ],
        tensions=[
            ContradictionPair(
                finding_a_id="F001", finding_b_id="F004", relation="contradicts",
                score=0.9, explanation="Opposite attention findings.", resolved=False
            ),
            ContradictionPair(
                finding_a_id="F001", finding_b_id="F002", relation="methodological_divergence",
                score=0.8, explanation="Different methods lead to different outcomes.", resolved=False
            ),
            ContradictionPair(
                finding_a_id="F001", finding_b_id="F003", relation="partial_consensus",
                score=0.7, explanation="Some overlap.", resolved=True
            )
        ]
    )

def test_synthesis_writer_valid(base_state):
    # TEST 1: Valid findings + valid tensions -> accepted
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="# Synthesis Report\n\n## Attention\nThe first study found improved attention. [F001]\n\n## Memory\nThe second study found decreased memory. [F002]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response) as mock_llm:
        report = writer.synthesize(base_state)
        mock_llm.assert_called_once()
        assert report == mock_response.markdown_report

def test_synthesis_writer_invalid_finding_id(base_state):
    # TEST 2: Generated report contains an invalid finding ID -> rejected.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="# Synthesis Report\nThis sentence has an invalid ID. [F999]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match=r"Grounding validation failed[\s\S]*Invalid IDs: \['F999'\]"):
            writer.synthesize(base_state)

def test_synthesis_writer_ungrounded_sentence(base_state):
    # TEST 3: Generated report contains a narrative sentence without finding ID -> rejected.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="# Synthesis Report\nThis is a narrative sentence without any citation.")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match=r"Grounding validation failed[\s\S]*Ungrounded Sentences: \['This is a narrative sentence without any citation.'\]"):
            writer.synthesize(base_state)

def test_synthesis_writer_multiple_valid_ids(base_state):
    # TEST 4: Generated report contains multiple valid finding IDs -> accepted.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="# Synthesis Report\nMultiple citations support this. [F001] [F002]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        report = writer.synthesize(base_state)
        assert report == mock_response.markdown_report

def test_synthesis_writer_no_tensions(base_state):
    # TEST 5: No tensions -> synthesis still works.
    base_state.tensions = []
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="This works without tensions. [F001]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        report = writer.synthesize(base_state)
        assert report == mock_response.markdown_report
        prompt = writer._build_prompt(base_state)
        assert "Tensions in this Subtopic:" not in prompt

def test_synthesis_writer_multiple_subtopics_grouping(base_state):
    # TEST 6: Multiple subtopics -> findings are grouped correctly.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "### Subtopic: Attention" in prompt
    assert "### Subtopic: Memory" in prompt

    attention_idx = prompt.find("### Subtopic: Attention")
    memory_idx = prompt.find("### Subtopic: Memory")
    f001_idx = prompt.find("- Finding ID: F001")
    f002_idx = prompt.find("- Finding ID: F002")

    assert attention_idx < f001_idx < memory_idx
    assert memory_idx < f002_idx

def test_synthesis_writer_tension_relations(base_state):
    # TEST 7: Contradiction relation is reflected in the actual LLM synthesis context.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "Relation: methodological_divergence" in prompt
    assert "Relation: contradicts" in prompt

def test_synthesis_writer_resolved_tension(base_state):
    # TEST 8: Resolved tension is represented correctly.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "Resolved: True" in prompt

def test_synthesis_writer_unresolved_tension(base_state):
    # TEST 9: Unresolved tension is represented correctly.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "Resolved: False" in prompt

def test_synthesis_writer_no_findings():
    # TEST 10: No findings -> safe failure and LLM is NOT called.
    empty_state = CycleState(query="query", subtopics=[])
    writer = SynthesisWriter()
    with patch("agents.synthesis_writer.generate_json") as mock_llm:
        with pytest.raises(ValueError, match="Insufficient evidence: No findings available to synthesize."):
            writer.synthesize(empty_state)
        mock_llm.assert_not_called()

def test_synthesis_writer_unassigned_finding(base_state):
    # TEST 11: Unassigned finding appears under "Other / Unassigned Evidence".
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "### Subtopic: Other / Unassigned Evidence" in prompt
    other_idx = prompt.find("### Subtopic: Other / Unassigned Evidence")
    f003_idx = prompt.find("- Finding ID: F003")
    assert f003_idx > other_idx

def test_synthesis_writer_malformed_citation(base_state):
    # TEST 12: Malformed citation is detected and causes validation failure.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="This citation is malformed. [F@001]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match=r"Grounding validation failed[\s\S]*Malformed Citations: \['F@001'\]"):
            writer.synthesize(base_state)

def test_synthesis_writer_short_ungrounded_sentence(base_state):
    # TEST 13: Short narrative/evidence sentence without citation is rejected.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="Short claim.")  # No exemption for < 3 words
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match=r"Grounding validation failed[\s\S]*Ungrounded Sentences: \['Short claim.'\]"):
            writer.synthesize(base_state)

def test_synthesis_writer_empty_response(base_state):
    # TEST 14: Empty LLM response is handled safely.
    writer = SynthesisWriter()
    with patch("agents.synthesis_writer.generate_json", return_value=SynthesisResponse(markdown_report="   ")):
        with pytest.raises(ValueError, match="LLM returned an empty response."):
            writer.synthesize(base_state)

def test_synthesis_writer_llm_exception(base_state):
    # TEST 15: LLM/API exception is handled safely.
    writer = SynthesisWriter()
    with patch("agents.synthesis_writer.generate_json", side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="API Error"):
            writer.synthesize(base_state)

def test_synthesis_writer_cycle_number(base_state):
    # TEST 16: Current cycle number is included in synthesis context.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert f"Current Research Cycle: {base_state.cycle_n}" in prompt

def test_synthesis_writer_fields_preserved(base_state):
    # TEST 17: All important finding fields are preserved in the generated prompt.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "- Finding ID: F001" in prompt
    assert "Claim: Improved attention." in prompt
    assert "Method: RCT" in prompt
    assert "Dataset: D1" in prompt
    assert "Metric/Value: Score / High" in prompt
    assert "Limitations: None" in prompt
    assert "Section Source: Results" in prompt

def test_synthesis_writer_tension_associated_ids(base_state):
    # TEST 18: Tension references are associated with the appropriate finding IDs.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)
    assert "- Finding A: F001 | Finding B: F002" in prompt
    assert "- Finding A: F001 | Finding B: F004" in prompt
    assert "- Finding A: F001 | Finding B: F003" in prompt

def test_synthesis_writer_abbreviations(base_state):
    # TEST 19: Abbreviations like e.g., i.e. do not split the sentence prematurely.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="Many metrics (e.g. score, i.e. time) improved. [F001]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        # Should not raise validation error because the citation belongs to the single sentence
        writer.synthesize(base_state)

def test_synthesis_writer_multiple_sentences_in_line(base_state):
    # TEST 20: Multiple sentences in a line are handled. The first fails because it has no citation.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="Evidence suggests improved attention. However, the effect varies across studies. [F001] [F002]")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match=r"Grounding validation failed[\s\S]*Ungrounded Sentences: \['Evidence suggests improved attention.'\]"):
            writer.synthesize(base_state)

def test_synthesis_writer_bulleted_lists(base_state):
    # TEST 21: Bullets are preserved correctly and evaluated.
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="- First finding. [F001]\n- Second finding without citation.")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match=r"Grounding validation failed[\s\S]*Ungrounded Sentences: \['- Second finding without citation.'\]"):
            writer.synthesize(base_state)

def test_synthesis_writer_subtopic_tension_association(base_state):
    # TEST 22: Tensions are associated with relevant subtopics based on finding papers.
    writer = SynthesisWriter()
    prompt = writer._build_prompt(base_state)

    # F001 and F004 both in Attention
    attention_idx = prompt.find("### Subtopic: Attention")
    memory_idx = prompt.find("### Subtopic: Memory")
    tension_f1_f4_idx = prompt.find("- Finding A: F001 | Finding B: F004")

    # Should be inside Attention subtopic
    assert attention_idx < tension_f1_f4_idx < memory_idx

    # F001 (Attention) and F002 (Memory) have no common subtopic -> Cross-Subtopic
    cross_idx = prompt.find("### Cross-Subtopic Tensions")
    tension_f1_f2_idx = prompt.find("- Finding A: F001 | Finding B: F002")
    assert cross_idx < tension_f1_f2_idx

    # F001 (Attention) and F003 (Unassigned) have no common subtopic -> Cross-Subtopic
    tension_f1_f3_idx = prompt.find("- Finding A: F001 | Finding B: F003")
    assert cross_idx < tension_f1_f3_idx

def test_synthesis_writer_valid_citations_after_punctuation(base_state):
    writer = SynthesisWriter()
    mock_responses = [
        "The study found improved attention. [F001]",
        "The study found improved attention! [F001]",
        "The study found improved attention? [F001]",
        "The study found improved attention. [F001] [F002]"
    ]
    for resp in mock_responses:
        with patch("agents.synthesis_writer.generate_json", return_value=SynthesisResponse(markdown_report=resp)):
            # Should not raise exception
            writer.synthesize(base_state)

def test_synthesis_writer_citation_before_sentence_fails(base_state):
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="[F001] The study found improved attention.")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match="Grounding validation failed"):
            writer.synthesize(base_state)

def test_synthesis_writer_citation_in_middle_fails(base_state):
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="The study [F001] found improved attention.")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match="Grounding validation failed"):
            writer.synthesize(base_state)

def test_synthesis_writer_completely_uncited_sentence(base_state):
    writer = SynthesisWriter()
    mock_response = SynthesisResponse(markdown_report="The study found improved attention.")
    with patch("agents.synthesis_writer.generate_json", return_value=mock_response):
        with pytest.raises(ValueError, match="Grounding validation failed"):
            writer.synthesize(base_state)
