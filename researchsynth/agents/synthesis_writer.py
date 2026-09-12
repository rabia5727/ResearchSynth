import re
from typing import Dict, List, Any
from pydantic import BaseModel

from researchsynth.state.schemas import CycleState, ExtractedFinding, ContradictionPair
from researchsynth.tools.llm import generate_json

class SynthesisResponse(BaseModel):
    markdown_report: str

class SynthesisWriter:
    def synthesize(self, state: CycleState) -> str:
        """
        Generates a synthesized Markdown report based on the CycleState.
        """
        if not state.findings:
            raise ValueError("Insufficient evidence: No findings available to synthesize.")

        prompt = self._build_prompt(state)

        # Use centralized generate_json abstraction
        response_dict = generate_json(prompt, schema=SynthesisResponse)
        report = response_dict.get("markdown_report", "")

        if not report or not report.strip():
            raise ValueError("LLM returned an empty response.")

        validation_result = self.validate_grounding(report, state.findings)
        if not validation_result["is_valid"]:
            error_msg = f"Grounding validation failed.\nInvalid IDs: {validation_result['invalid_finding_ids']}\nMalformed Citations: {validation_result['malformed_citations']}\nUngrounded Sentences: {validation_result['ungrounded_sentences']}"
            raise ValueError(error_msg)

        return report

    def _build_prompt(self, state: CycleState) -> str:
        """
        Constructs the prompt for the LLM based on grouped findings and tensions.
        """
        # Group findings by subtopic
        grouped_findings: Dict[str, List[ExtractedFinding]] = {subtopic: [] for subtopic in state.subtopics}
        grouped_findings["Other / Unassigned Evidence"] = []

        finding_to_subtopics: Dict[str, List[str]] = {}

        for finding in state.findings:
            assigned = False
            paper = next((p for p in state.papers if p.id == finding.paper_id), None)
            finding_subs = []

            if paper:
                for subtopic in state.subtopics:
                    if subtopic in paper.subtopic_tags:
                        grouped_findings[subtopic].append(finding)
                        finding_subs.append(subtopic)
                        assigned = True

            if not assigned:
                grouped_findings["Other / Unassigned Evidence"].append(finding)
                finding_subs.append("Other / Unassigned Evidence")

            finding_to_subtopics[finding.id] = finding_subs

        # Group tensions by subtopic
        grouped_tensions: Dict[str, List[ContradictionPair]] = {subtopic: [] for subtopic in grouped_findings.keys()}
        grouped_tensions["Cross-Subtopic Tensions"] = []

        for tension in state.tensions:
            subs_a = finding_to_subtopics.get(tension.finding_a_id, ["Other / Unassigned Evidence"])
            subs_b = finding_to_subtopics.get(tension.finding_b_id, ["Other / Unassigned Evidence"])

            common_subs = set(subs_a).intersection(subs_b)
            if common_subs:
                for st in common_subs:
                    grouped_tensions[st].append(tension)
            else:
                grouped_tensions["Cross-Subtopic Tensions"].append(tension)

        # Build the prompt
        prompt_lines = [
            f"Research Question: {state.query}",
            f"Current Research Cycle: {state.cycle_n}",
            "Write a synthesized Markdown report answering the research question based on the following evidence.",
            "IMPORTANT INSTRUCTIONS:",
            "- Organize the report by subtopic using Markdown headings (e.g. ## Subtopic).",
            "- Synthesize evidence rather than summarizing papers one-by-one.",
            "- Explicitly discuss contradictions, partial consensus, and methodological divergences.",
            "- Identify limitations of the evidence.",
            "- Every sentence containing an evidence claim must carry a [finding_id] marker at the end of the sentence.",
            "- Use ONLY the supplied findings. Do not invent facts or paper details.",
            "- Do not output ungrounded claims.",
            "\nEVIDENCE BY SUBTOPIC:"
        ]

        for subtopic, findings in grouped_findings.items():
            tensions = grouped_tensions.get(subtopic, [])
            if findings or tensions:
                prompt_lines.append(f"\n### Subtopic: {subtopic}\n")

                if findings:
                    prompt_lines.append("#### Findings:")
                    for f in findings:
                        prompt_lines.append(f"- Finding ID: {f.id}")
                        prompt_lines.append(f"  Claim: {f.claim}")
                        if f.method:
                            prompt_lines.append(f"  Method: {f.method}")
                        if f.dataset:
                            prompt_lines.append(f"  Dataset: {f.dataset}")
                        if f.metric or f.value:
                            prompt_lines.append(f"  Metric/Value: {f.metric} / {f.value}")
                        if f.limitations:
                            prompt_lines.append(f"  Limitations: {f.limitations}")
                        prompt_lines.append(f"  Section Source: {f.section_source}")

                if tensions:
                    prompt_lines.append("\n#### Tensions in this Subtopic:")
                    for t in tensions:
                        prompt_lines.append(f"- Finding A: {t.finding_a_id} | Finding B: {t.finding_b_id}")
                        prompt_lines.append(f"  Relation: {t.relation}")
                        prompt_lines.append(f"  Score: {t.score}")
                        prompt_lines.append(f"  Explanation: {t.explanation}")
                        prompt_lines.append(f"  Resolved: {t.resolved}")

        cross_tensions = grouped_tensions.get("Cross-Subtopic Tensions", [])
        if cross_tensions:
            prompt_lines.append("\n### Cross-Subtopic Tensions\n")
            for t in cross_tensions:
                prompt_lines.append(f"- Finding A: {t.finding_a_id} | Finding B: {t.finding_b_id}")
                prompt_lines.append(f"  Relation: {t.relation}")
                prompt_lines.append(f"  Score: {t.score}")
                prompt_lines.append(f"  Explanation: {t.explanation}")
                prompt_lines.append(f"  Resolved: {t.resolved}")

        return "\n".join(prompt_lines)

    def validate_grounding(self, report: str, valid_findings: List[ExtractedFinding]) -> Dict[str, Any]:
        """
        Validates that every narrative sentence in the report contains at least one valid finding ID.
        """
        valid_ids = {f.id for f in valid_findings}
        lines = report.split('\n')

        invalid_finding_ids = set()
        ungrounded_sentences = []
        malformed_citations = []
        grounded_sentences_count = 0
        total_sentences_count = 0

        # Split sentences dynamically avoiding abbreviations
        # Sentence boundary: .!? optionally followed by spaces/citations, then a space, then alphanumeric.
        split_pattern = r'(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)(?<!\betc\.)(?<!\bal\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bFig\.)(?<=[.!?\]])\s+(?=[A-Za-z0-9])'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip Markdown headings
            if line.startswith('#'):
                continue

            sentences = [s.strip() for s in re.split(split_pattern, line) if s.strip()]

            for sentence in sentences:
                clean_sentence = re.sub(r'^[-*0-9.]*\s*', '', sentence).strip()

                if not clean_sentence:
                    continue

                total_sentences_count += 1

                # Check for citations strictly after the sentence-ending punctuation
                end_citations_match = re.search(r'[.!?]\s*((?:\[[^\]]+\]\s*)+)$', sentence)

                if end_citations_match:
                    end_matches = re.findall(r'\[([^\]]+)\]', end_citations_match.group(1))
                    text_without_end = sentence[:end_citations_match.start()]
                    middle_matches = re.findall(r'\[([^\]]+)\]', text_without_end)
                else:
                    end_matches = []
                    middle_matches = re.findall(r'\[([^\]]+)\]', sentence)

                has_valid_end_id = False
                for match in end_matches:
                    if match in valid_ids:
                        has_valid_end_id = True
                    else:
                        if re.match(r'^[A-Za-z0-9\-_]+$', match):
                            invalid_finding_ids.add(match)
                        else:
                            malformed_citations.append(match)

                for match in middle_matches:
                    # Citations in the middle invalidate the sentence's grounding
                    has_valid_end_id = False
                    if match not in valid_ids:
                        if re.match(r'^[A-Za-z0-9\-_]+$', match):
                            invalid_finding_ids.add(match)
                        else:
                            malformed_citations.append(match)

                if not has_valid_end_id:
                    ungrounded_sentences.append(sentence)
                else:
                    grounded_sentences_count += 1

        is_valid = len(invalid_finding_ids) == 0 and len(ungrounded_sentences) == 0 and len(malformed_citations) == 0

        return {
            "is_valid": is_valid,
            "invalid_finding_ids": list(invalid_finding_ids),
            "ungrounded_sentences": ungrounded_sentences,
            "malformed_citations": malformed_citations,
            "total_sentences": total_sentences_count,
            "grounded_sentences": grounded_sentences_count
        }
